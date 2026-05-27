"""
Pipe-based frame I/O: PipeFrameSource, PipeFrameSink, ControlChannel.

These speak the integration protocol with the Electron host (see
docs/integration-protocol.md). The underlying transport (TCP for
the spike; Windows named pipes later in production) is abstracted
behind stitcher.transport.Transport -- this module only knows about
the protocol bytes.

Pixel conversion note: the wire format is RGBA8888 (matches
Electron's HTML canvas / ImageData), but the rest of VideoStitcher
operates on OpenCV-native BGR uint8 arrays. PipeFrameSource
converts RGBA -> BGR on input; PipeFrameSink converts BGR -> RGBA
on output. The conversion is a numpy reindex (no copy of memory
beyond one pass) so the cost is irrelevant compared to YOLOE +
warping + DP.
"""

import numpy as np

from stitcher.frame_io import FrameSink, FrameSource
from stitcher.protocol import (
    FORMAT_RGBA8888,
    HEADER_SIZE,
    OUTPUT_CAMERA_INDEX,
    ProtocolError,
    decode_control_message,
    encode_control_message,
    make_log,
    pack_frame_header,
    unpack_frame_header,
)


class PipeFrameSource(FrameSource):
    """
    Reads paired (left, right) frames off the frames channel.

    The Electron host sends frames interleaved by camera (one frame
    per camera, alternating). This class buffers one frame per
    camera and emits a paired (frame_left_bgr, frame_right_bgr,
    timestamp_us) tuple when both arrive.

    On wire-format errors, raises ProtocolError. On transport
    disconnect (i.e. session ended), read_pair() returns None.
    """

    def __init__(self, frames_transport, control_transport=None,
                 cam_left_index=0, cam_right_index=1,
                 declared_fps=30.0):
        self._fr = frames_transport
        self._ctrl = control_transport
        self._cam_left = cam_left_index
        self._cam_right = cam_right_index
        # Most recent frame per camera, awaiting its pair.
        self._pending = {cam_left_index: None, cam_right_index: None}
        # Nominal fps used by the pipeline for EMA alphas + FG
        # recompute cadence. In pipe mode there's no file-fps to read;
        # the host can declare a value via start_session (the protocol
        # doesn't currently carry one per camera, so this is fed in
        # via the constructor). Spike default: 30 fps.
        self._declared_fps = float(declared_fps)

    def open(self):
        # Transports are passed in already connected (the protocol
        # handshake runs before this source is instantiated), so
        # there's nothing to open at this layer. The method exists
        # to satisfy the FrameSource ABC contract.
        pass

    def close(self):
        # The two transports are owned by pipe_main (which manages
        # the session lifecycle and closes them in its finally
        # block), so this is a no-op. Kept to satisfy the ABC.
        pass

    @property
    def output_fps(self):
        return self._declared_fps

    def summary(self):
        return (f"[pipe] PipeFrameSource: declared {self._declared_fps:.2f} "
                f"fps, paired cams ({self._cam_left}, {self._cam_right})")

    def summary_post(self):
        # No file-mode "frames dropped because of FPS desync" stat
        # to report; the host is responsible for pairing.
        return "[pipe] PipeFrameSource finished."

    def read_pair(self):
        """
        Block until one frame from each camera is available, then
        return (frame_left_bgr, frame_right_bgr, timestamp_us).
        Returns None on transport disconnect (session ended).
        """
        while True:
            try:
                header_bytes = self._fr.read_exact(HEADER_SIZE)
            except ConnectionError:
                return None

            header = unpack_frame_header(header_bytes)

            if header["format"] != FORMAT_RGBA8888:
                raise ProtocolError(
                    f"frame_format: only RGBA8888 (1) is supported, "
                    f"got {header['format']}"
                )

            try:
                payload = self._fr.read_exact(header["payload_length"])
            except ConnectionError:
                return None

            expected = header["width"] * header["height"] * 4
            if header["payload_length"] != expected:
                raise ProtocolError(
                    f"frame_length: header says {header['payload_length']}, "
                    f"expected w*h*4 = {expected}"
                )

            # RGBA bytes -> BGR uint8 HxWx3 (drop alpha).
            arr = np.frombuffer(payload, dtype=np.uint8)
            arr = arr.reshape(header["height"], header["width"], 4)
            # arr[..., [2, 1, 0]] reads channels R(0), G(1), B(2)
            # from RGBA and arranges them as BGR. .copy() detaches
            # from the shared payload buffer.
            frame_bgr = arr[..., [2, 1, 0]].copy()

            cam = header["camera_index"]
            if cam not in self._pending:
                self._log_warn(
                    f"unknown camera_index {cam}; dropping frame"
                )
                continue

            self._pending[cam] = (frame_bgr, header["timestamp_us"])

            left = self._pending[self._cam_left]
            right = self._pending[self._cam_right]
            if left is not None and right is not None:
                self._pending[self._cam_left] = None
                self._pending[self._cam_right] = None
                # Use the later of the two timestamps as the pair's
                # timestamp -- that's the moment by which both
                # cameras had captured.
                ts = max(left[1], right[1])
                return left[0], right[0], ts

    def _log_warn(self, message):
        if self._ctrl is None:
            return
        try:
            self._ctrl.write(encode_control_message(
                make_log("warn", message)
            ))
        except ConnectionError:
            pass


class PipeFrameSink(FrameSink):
    """
    Writes stitched output frames to the frames channel.

    Takes a BGR uint8 HxWx3 numpy array (the pipeline's native
    format), converts to RGBA, packs the 32-byte header, sends both
    header and payload over the transport. Async writes (GPU path)
    are handled by synchronising the cuda event on the calling
    thread, running the post-sync transform (debug overlays /
    tracking crop), then sending. This serialises encoding on the
    composite worker -- acceptable for the spike; a ThreadedVideoWriter-
    style background thread can be added later if it becomes a
    bottleneck.
    """

    def __init__(self, frames_transport):
        self._fr = frames_transport

    def open(self, width, height, fps):
        # The transport is already connected and the wire format is
        # self-describing per frame, so there's no per-session
        # setup. Kept to satisfy the FrameSink ABC.
        pass

    def write(self, frame_bgr, timestamp_us=0):
        self._send_bgr(frame_bgr, timestamp_us)

    def write_async(self, pinned, event, post_sync_fn=None, free_cb=None,
                    timestamp_us=0):
        try:
            # Wait for the GPU pyramid + pinned copy to finish.
            event.synchronize()
            arr = pinned.numpy()
            if post_sync_fn is not None:
                arr = post_sync_fn(arr)
            self._send_bgr(arr, timestamp_us)
        finally:
            if free_cb is not None:
                free_cb()

    def close(self):
        # Transport is owned by pipe_main; no-op here.
        pass

    def _send_bgr(self, frame_bgr, timestamp_us):
        H, W = frame_bgr.shape[:2]
        # BGR -> RGBA. In a BGR array, ch 0 = B, 1 = G, 2 = R.
        # In an RGBA array,         ch 0 = R, 1 = G, 2 = B, 3 = A.
        # So we swap channels 0 and 2 and add a fully-opaque alpha.
        rgba = np.empty((H, W, 4), dtype=np.uint8)
        rgba[..., 0] = frame_bgr[..., 2]  # R
        rgba[..., 1] = frame_bgr[..., 1]  # G
        rgba[..., 2] = frame_bgr[..., 0]  # B
        rgba[..., 3] = 255

        payload = rgba.tobytes()
        header = pack_frame_header(
            camera_index=OUTPUT_CAMERA_INDEX,
            timestamp_us=int(timestamp_us),
            width=W, height=H,
            payload_length=len(payload),
        )
        try:
            self._fr.write(header)
            self._fr.write(payload)
        except ConnectionError:
            # Transport closed mid-write; treat as session-end.
            pass


class ControlChannel:
    """
    Thin wrapper over the control transport: send/recv dict messages.

    Both Electron and Python sides use this. Send dicts, receive
    dicts (or None on disconnect). The dict shape is defined by the
    protocol-doc message types in stitcher.protocol.
    """

    def __init__(self, transport):
        self._t = transport

    def send(self, msg):
        """Serialize and write one message (line-delimited JSON)."""
        self._t.write(encode_control_message(msg))

    def recv(self):
        """
        Block until one message arrives. Returns the parsed dict, or
        None if the peer disconnected. Raises ProtocolError on
        malformed JSON.
        """
        try:
            line = self._t.read_line()
        except ConnectionError:
            return None
        return decode_control_message(line)

    def close(self):
        self._t.close()
