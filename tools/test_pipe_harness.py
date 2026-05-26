"""
End-to-end test harness for the pipe protocol (no real stitching).

Goal: exercise every section of docs/integration-protocol.md with
real code before we wire pipe mode into the actual stitching
pipeline. If this harness passes, the protocol implementation
itself is solid -- subsequent work just plugs the real pipeline
into the source/sink shaped by this exercise.

What this script does:

  - Plays BOTH roles in a single Python process, in two threads.
    The transport between them is two TCP sockets on localhost
    (same byte protocol as the eventual Windows named pipes).

      "host"      = Electron stand-in. Listens on two ports,
                    accepts the stitcher's connection on each,
                    sends hello + start_session, generates N
                    paired RGBA test frames, reads back N stitched
                    frames, sends stop_session + shutdown.

      "stitcher"  = VideoStitcher stand-in. Connects to both ports,
                    exchanges hello, runs a frame pump that pairs
                    incoming frames and produces a FAKE stitched
                    output (side-by-side horizontal concat -- no
                    real algorithm). Returns it on the same channel.

  - Checks all the protocol invariants along the way:
      hello / hello_ack version negotiation
      start_session / session_started
      32-byte frame header round-trip
      RGBA -> BGR -> RGBA pixel round-trip
      stop_session / session_stopped
      shutdown / clean exit

  - Prints PASS / FAIL.

Run:
    python tools/test_pipe_harness.py
    python tools/test_pipe_harness.py --n_frames 20

Once this prints PASS, we know the wire protocol works on this
machine; next step is to plug the real VideoStitcher pipeline in
behind PipeFrameSource / PipeFrameSink.
"""

import argparse
import os
import sys
import threading
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402

from stitcher.pipe_io import (  # noqa: E402
    ControlChannel, PipeFrameSink, PipeFrameSource,
)
from stitcher.protocol import (  # noqa: E402
    FORMAT_RGBA8888,
    HEADER_SIZE,
    OUTPUT_CAMERA_INDEX,
    PROTOCOL_VERSION,
    make_hello,
    make_hello_ack,
    make_session_started,
    make_session_stopped,
    make_shutdown,
    make_start_session,
    make_stop_session,
    pack_frame_header,
    unpack_frame_header,
)
from stitcher.transport import TCPListener, TCPTransport  # noqa: E402


TEST_WIDTH = 320
TEST_HEIGHT = 240


def _make_test_frame_rgba(cam_idx, frame_idx,
                          width=TEST_WIDTH, height=TEST_HEIGHT):
    """Build a small RGBA test pattern. Cam 0 is reddish, cam 1
    blueish; a scrolling yellow band lets us eyeball that frames
    don't get swapped."""
    img = np.zeros((height, width, 4), dtype=np.uint8)
    if cam_idx == 0:
        img[..., 0] = 200  # R
    else:
        img[..., 2] = 200  # B
    band_x = (frame_idx * 20) % width
    img[:, band_x:band_x + 10, 0:3] = [255, 255, 0]  # yellow
    img[..., 3] = 255
    return img


def host_thread(control_listener, frames_listener, n_frames, result):
    """Electron stand-in: server side of both channels."""
    log = lambda m: print(f"[host]     {m}", flush=True)
    try:
        log("waiting for stitcher to connect to control channel...")
        ctrl_t = control_listener.accept()
        log("waiting for stitcher to connect to frames channel...")
        frames_t = frames_listener.accept()
        log("stitcher connected on both channels")

        ctrl = ControlChannel(ctrl_t)

        # 1. hello / hello_ack
        log("sending hello")
        ctrl.send(make_hello(versions=[PROTOCOL_VERSION]))
        ack = ctrl.recv()
        log(f"got hello_ack: {ack}")
        assert ack is not None and ack["type"] == "hello_ack", ack
        assert ack["protocol_version"] == PROTOCOL_VERSION, ack

        # 2. start_session
        log("sending start_session")
        ctrl.send(make_start_session(input_cameras=[
            {"index": 0, "label": "left",
             "width": TEST_WIDTH, "height": TEST_HEIGHT},
            {"index": 1, "label": "right",
             "width": TEST_WIDTH, "height": TEST_HEIGHT},
        ]))
        started = ctrl.recv()
        log(f"got session_started: {started}")
        assert started and started["type"] == "session_started", started
        assert started["output_width"] == TEST_WIDTH * 2, started
        assert started["output_height"] == TEST_HEIGHT, started

        # 3 + 4. Send N paired frames AND read N stitched outputs.
        # Must happen concurrently: each 320x240 RGBA frame is ~300KB,
        # larger than the default ~64KB TCP send buffer. If the host
        # only sends and the stitcher only sends back, both sides
        # block on sendall after the first frame -- classic
        # bidirectional blocking-IO deadlock. Output reader runs on
        # its own thread; input sender stays on the main host thread.
        log(f"sending {n_frames} paired frames + reading "
            f"{n_frames} stitched outputs (concurrent)")
        received_box = [0]
        last_payload_box = [None]
        last_hdr_box = [None]
        reader_err = [None]

        def read_outputs():
            try:
                for _fidx in range(n_frames):
                    hdr_bytes = frames_t.read_exact(HEADER_SIZE)
                    hdr = unpack_frame_header(hdr_bytes)
                    assert hdr["camera_index"] == OUTPUT_CAMERA_INDEX, hdr
                    assert hdr["width"] == TEST_WIDTH * 2, hdr
                    assert hdr["height"] == TEST_HEIGHT, hdr
                    payload = frames_t.read_exact(hdr["payload_length"])
                    assert len(payload) == hdr["width"] * hdr["height"] * 4
                    received_box[0] += 1
                    last_payload_box[0] = payload
                    last_hdr_box[0] = hdr
            except Exception as e:
                reader_err[0] = e

        reader = threading.Thread(target=read_outputs, daemon=True)
        reader.start()

        base_ts = int(time.time() * 1e6)
        for fidx in range(n_frames):
            ts = base_ts + fidx * 33_333  # ~30 fps
            for cam in (0, 1):
                img = _make_test_frame_rgba(cam, fidx)
                payload = img.tobytes()
                header = pack_frame_header(
                    camera_index=cam, timestamp_us=ts,
                    width=TEST_WIDTH, height=TEST_HEIGHT,
                    payload_length=len(payload),
                    fmt=FORMAT_RGBA8888,
                )
                frames_t.write(header + payload)

        reader.join(timeout=15)
        if reader_err[0] is not None:
            raise reader_err[0]
        received = received_box[0]
        payload = last_payload_box[0]
        hdr = last_hdr_box[0]
        log(f"received {received}/{n_frames} stitched frames")

        # Sanity: round-trip a single pixel. The fake stitcher does
        # side-by-side concat, so the left half of the output's left
        # pixel should be reddish (from cam 0 input).
        # Convert the last received payload to RGBA HxWx4.
        last_rgba = np.frombuffer(payload, dtype=np.uint8).reshape(
            hdr["height"], hdr["width"], 4,
        )
        # Sample a pixel away from the scrolling yellow band, in the
        # cam-0 (left) half.
        sample = last_rgba[hdr["height"] // 2, 5]  # (R, G, B, A)
        assert sample[0] > 100, f"left half should be reddish, got {sample}"
        assert sample[3] == 255, f"alpha should be 255, got {sample}"

        # 5. stop_session
        log("sending stop_session")
        ctrl.send(make_stop_session())
        stopped = ctrl.recv()
        log(f"got session_stopped: {stopped}")
        assert stopped and stopped["type"] == "session_stopped", stopped

        # 6. shutdown
        log("sending shutdown")
        ctrl.send(make_shutdown())

        # Give the peer a moment to close cleanly.
        time.sleep(0.1)

        result["host_ok"] = True
        result["host_frames_received"] = received
    except Exception as e:
        result["host_error"] = repr(e)
        import traceback
        result["host_traceback"] = traceback.format_exc()
        log(f"ERROR: {e}")


def stitcher_thread(control_port, frames_port, result):
    """VideoStitcher stand-in: client side of both channels."""
    log = lambda m: print(f"[stitcher] {m}", flush=True)
    try:
        log("connecting to host control channel...")
        ctrl_t = TCPTransport("127.0.0.1", control_port)
        log("connecting to host frames channel...")
        frames_t = TCPTransport("127.0.0.1", frames_port)
        log("connected to both channels")

        ctrl = ControlChannel(ctrl_t)

        # 1. hello
        hello = ctrl.recv()
        log(f"got hello: {hello}")
        assert hello and hello["type"] == "hello"
        versions = hello.get("protocol_versions", [])
        assert PROTOCOL_VERSION in versions, versions
        ctrl.send(make_hello_ack(stitcher_version="0.0.0-spike"))

        # 2. start_session
        start = ctrl.recv()
        log(f"got start_session ({len(start.get('input_cameras', []))} cameras)")
        assert start and start["type"] == "start_session"
        cams = start["input_cameras"]
        W = cams[0]["width"]
        H = cams[0]["height"]
        out_w = W * 2  # fake stitcher: side-by-side concat
        out_h = H
        ctrl.send(make_session_started(out_w, out_h))

        # 3. Frame pump runs in a thread so we can listen on control
        # for stop_session in parallel.
        source = PipeFrameSource(frames_t, ctrl_t)
        sink = PipeFrameSink(frames_t)
        stop_event = threading.Event()
        frame_count = [0]

        def pump():
            log("frame pump started")
            while not stop_event.is_set():
                pair = source.read_pair()
                if pair is None:
                    log("source disconnected")
                    return
                left, right, ts = pair
                # Fake stitching: side-by-side horizontal concat.
                stitched = np.concatenate([left, right], axis=1)
                sink.write(stitched, ts)
                frame_count[0] += 1

        pump_t = threading.Thread(target=pump)
        pump_t.start()

        # 4. Wait for stop_session / shutdown
        while True:
            msg = ctrl.recv()
            if msg is None:
                log("control disconnected")
                break
            t = msg.get("type")
            log(f"control rx: {t}")
            if t == "stop_session":
                # In a real pipeline we'd drain in-flight work first.
                # Here the host has already read every output before
                # sending stop, so the pump is idle on the next read.
                # Close the frames channel to unblock it.
                stop_event.set()
                frames_t.close()
                pump_t.join(timeout=2.0)
                ctrl.send(make_session_stopped())
            elif t == "shutdown":
                break

        ctrl_t.close()
        result["stitcher_ok"] = True
        result["stitcher_frames_emitted"] = frame_count[0]
        log(f"done, emitted {frame_count[0]} stitched frames")
    except Exception as e:
        result["stitcher_error"] = repr(e)
        import traceback
        result["stitcher_traceback"] = traceback.format_exc()
        log(f"ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_frames", type=int, default=5,
                        help="paired frames to push through (default 5)")
    args = parser.parse_args()

    control_listener = TCPListener("127.0.0.1", 0)
    frames_listener = TCPListener("127.0.0.1", 0)
    cp = control_listener.port
    fp = frames_listener.port
    print(f"[main]     control port {cp}, frames port {fp}")

    result = {}
    host_t = threading.Thread(
        target=host_thread,
        args=(control_listener, frames_listener, args.n_frames, result),
    )
    stitch_t = threading.Thread(
        target=stitcher_thread, args=(cp, fp, result),
    )
    host_t.start()
    # Slight head-start so the host's listening sockets are ready
    # before the stitcher tries to connect.
    time.sleep(0.05)
    stitch_t.start()
    host_t.join(timeout=30)
    stitch_t.join(timeout=30)

    print()
    print(f"[main]     result keys: {sorted(result)}")
    if "host_error" in result:
        print(f"[main]     host_error: {result['host_error']}")
        print(result.get("host_traceback", ""))
    if "stitcher_error" in result:
        print(f"[main]     stitcher_error: {result['stitcher_error']}")
        print(result.get("stitcher_traceback", ""))

    ok = (
        result.get("host_ok") and result.get("stitcher_ok")
        and result.get("host_frames_received") == args.n_frames
        and result.get("stitcher_frames_emitted") == args.n_frames
    )
    if ok:
        print(f"[main]     PASS  ({args.n_frames} frames round-tripped)")
        return 0
    print("[main]     FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
