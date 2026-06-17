"""
Pipe-based frame I/O: PipeFrameSource, PipeFrameSink, ControlChannel.

These speak the integration protocol with the Electron host (see
docs/integration-protocol.md). The underlying transport (TCP for
the spike; Windows named pipes later in production) is abstracted
behind stitcher.transport.Transport -- this module only knows about
the protocol bytes.

Pixel conversion note: the wire format on the wire can be either
RGBA8888 or JPEG -- the renderer picks per-frame and stamps a
`format` byte in the header.

  RGBA8888 (1): raw RGBA bytes, used in dev / file mode. Decoded
                with a numpy reindex (RGBA -> BGR).
  JPEG     (2): JPEG-encoded bytes from the portal-agent's
                `canvas.toBlob('image/jpeg', q)`. Decoded with
                cv2.imdecode which produces a BGR array directly,
                so no channel swap is needed after decode. This
                path exists because raw RGBA at 1280x720 saturated
                the renderer-to-Python IPC throughput at ~5 fps;
                JPEG cuts the per-frame payload ~10x and unblocks
                the source rate.

The rest of VideoStitcher operates on OpenCV-native BGR uint8
arrays, so either decoder produces the same shape and dtype
downstream. PipeFrameSink converts BGR -> RGBA on output (the
return path is always RGBA -- there's no JPEG round-trip).
"""

import cv2
import numpy as np

from stitcher.frame_io import FrameSink, FrameSource
from stitcher.protocol import (
    FORMAT_JPEG,
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
    Reads N-camera tuples (N == 2 or 3) off the frames channel.

    The Electron host sends frames interleaved by camera (one frame
    per camera). This class buffers one frame per camera and emits a
    paired / triplet tuple when all expected cameras have arrived.

    On wire-format errors, raises ProtocolError. On transport
    disconnect (i.e. session ended), read_pair() / read_triplet()
    return None.

    For 2-camera mode (default), use read_pair(). For 3-camera mode,
    construct with cam_indices=(0, 1, 2) and call read_triplet().
    """

    def __init__(self, frames_transport, control_transport=None,
                 cam_left_index=0, cam_right_index=1,
                 cam_indices=None,
                 declared_fps=30.0):
        self._fr = frames_transport
        self._ctrl = control_transport
        # New explicit cam_indices argument supersedes the cam_left/right
        # pair for 3-camera mode. Default falls back to the 2-camera
        # (left, right) tuple so all existing 2-cam callers keep working
        # without changes.
        if cam_indices is None:
            cam_indices = (cam_left_index, cam_right_index)
        self._cam_indices = tuple(cam_indices)
        if len(self._cam_indices) not in (2, 3):
            raise ValueError(
                f"PipeFrameSource supports 2 or 3 cameras, got "
                f"{len(self._cam_indices)}"
            )
        # Most recent frame per camera, awaiting its pair / triplet.
        self._pending = {idx: None for idx in self._cam_indices}
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

    @property
    def n_cameras(self):
        return len(self._cam_indices)

    def summary(self):
        return (
            f"[pipe] PipeFrameSource: declared {self._declared_fps:.2f} "
            f"fps, cams {self._cam_indices}"
        )

    def summary_post(self):
        # No file-mode "frames dropped because of FPS desync" stat
        # to report; the host is responsible for pairing.
        return "[pipe] PipeFrameSource finished."

    def _read_next_frame_into_pending(self):
        """
        Block until ONE frame arrives off the wire. Decode, validate,
        and stash into self._pending under its camera_index. Returns
        True on success, False on transport disconnect.

        Shared between read_pair and read_triplet so the wire-decode
        logic isn't duplicated. The caller's loop decides when enough
        cameras have reported in to emit a tuple.
        """
        try:
            header_bytes = self._fr.read_exact(HEADER_SIZE)
        except ConnectionError:
            return False

        header = unpack_frame_header(header_bytes)
        fmt = header["format"]
        if fmt not in (FORMAT_RGBA8888, FORMAT_JPEG):
            raise ProtocolError(
                f"frame_format: unsupported format {fmt}; expected "
                f"RGBA8888 ({FORMAT_RGBA8888}) or JPEG ({FORMAT_JPEG})"
            )
        try:
            payload = self._fr.read_exact(header["payload_length"])
        except ConnectionError:
            return False
        if fmt == FORMAT_RGBA8888:
            # Sanity-check the payload size matches the geometry the
            # header advertises. JPEG payloads are encoder-dependent
            # so no equivalent check exists for that path.
            expected = header["width"] * header["height"] * 4
            if header["payload_length"] != expected:
                raise ProtocolError(
                    f"frame_length: header says {header['payload_length']}, "
                    f"expected w*h*4 = {expected}"
                )
            arr = np.frombuffer(payload, dtype=np.uint8)
            arr = arr.reshape(header["height"], header["width"], 4)
            # RGBA -> BGR via channel reindex. .copy() detaches from
            # the shared payload buffer.
            frame_bgr = arr[..., [2, 1, 0]].copy()
        else:  # FORMAT_JPEG
            # cv2.imdecode returns a fresh BGR uint8 array (already
            # detached from `payload`), shape (H, W, 3). No channel
            # reindex needed -- JPEG decode produces BGR natively
            # in OpenCV. Returns None on malformed bytes, which we
            # surface as a ProtocolError so the bridge sees the
            # disconnect-style cleanup rather than a confusing
            # downstream crash.
            arr = np.frombuffer(payload, dtype=np.uint8)
            frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise ProtocolError(
                    f"frame_jpeg_decode: cv2.imdecode failed on "
                    f"{header['payload_length']} bytes for camera "
                    f"{header['camera_index']}"
                )
        cam = header["camera_index"]
        if cam not in self._pending:
            self._log_warn(
                f"unknown camera_index {cam}; dropping frame"
            )
            return True  # not a disconnect, just an extra camera
        self._pending[cam] = (frame_bgr, header["timestamp_us"])
        return True

    def _all_pending_ready(self):
        return all(self._pending[idx] is not None for idx in self._cam_indices)

    def _drain_and_emit_tuple(self):
        """
        Move pending frames into an ordered tuple, reset pending,
        return (frame_0_bgr, ..., frame_N-1_bgr, ts_us). The ts is
        the max across cameras -- the moment by which all of them
        had captured.
        """
        frames = []
        timestamps = []
        for idx in self._cam_indices:
            frame, ts = self._pending[idx]
            frames.append(frame)
            timestamps.append(ts)
            self._pending[idx] = None
        return (*frames, max(timestamps))

    def read_pair(self):
        """
        Block until one frame from each of the 2 configured cameras
        is available, then return (frame_left_bgr, frame_right_bgr,
        timestamp_us). Returns None on transport disconnect.

        Raises ValueError if this source was configured for 3 cameras.
        """
        if len(self._cam_indices) != 2:
            raise ValueError(
                f"read_pair() called on a {len(self._cam_indices)}-camera "
                "source. Use read_triplet() instead."
            )
        while True:
            if not self._read_next_frame_into_pending():
                return None
            if self._all_pending_ready():
                return self._drain_and_emit_tuple()

    def read_triplet(self):
        """
        Block until one frame from each of the 3 configured cameras
        is available, then return (frame_left_bgr, frame_center_bgr,
        frame_right_bgr, timestamp_us). Returns None on transport
        disconnect.

        Raises ValueError if this source was configured for 2 cameras.
        """
        if len(self._cam_indices) != 3:
            raise ValueError(
                f"read_triplet() called on a {len(self._cam_indices)}-camera "
                "source. Use read_pair() instead."
            )
        while True:
            if not self._read_next_frame_into_pending():
                return None
            if self._all_pending_ready():
                return self._drain_and_emit_tuple()

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
