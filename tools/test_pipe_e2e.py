"""
End-to-end pipe-mode harness with the real stitching pipeline.

Spawns video_stitcher_seam_gpu.py with --io pipe and feeds it real
RGBA frames decoded from two mp4 files. Receives the stitched
output frames back over the same pipe and writes them to an mp4
the user can play.

The standalone-protocol-only harness (tools/test_pipe_harness.py)
validates that the wire format is implemented correctly using a
fake side-by-side concat as the stitcher. This script validates
that the protocol drives the REAL pipeline end-to-end.

Architecture mirrors what the Electron host will eventually do:
  1. Open two TCP listening sockets (control + frames).
  2. Spawn the stitcher as a child process pointed at those ports.
  3. Accept both connections, run the hello / start_session
     handshake.
  4. Concurrently:
       - send paired RGBA frames decoded from the input videos
       - listen for async control messages (session_started, log,
         session_stopped, error)
       - receive stitched RGBA frames and write them to an mp4
  5. Close the frames channel when done sending -> stitcher exits
     naturally; harness waits for the child process.

Usage:
    python tools/test_pipe_e2e.py \
        --video_a videos/sf_left.mp4 \
        --video_b videos/sf_right.mp4 \
        --output stitched_e2e.mp4 \
        --n_frames 20

Tip: --max_resolution 640 resizes input frames to fit comfortably
inside default Windows TCP buffers while developing. Drop it once
you want the full-res run.
"""

import argparse
import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from stitcher.pipe_io import ControlChannel  # noqa: E402
from stitcher.protocol import (  # noqa: E402
    FORMAT_RGBA8888,
    HEADER_SIZE,
    OUTPUT_CAMERA_INDEX,
    PROTOCOL_VERSION,
    make_hello,
    make_start_session,
    pack_frame_header,
    unpack_frame_header,
)
from stitcher.transport import TCPListener  # noqa: E402


def _resize_long_side(frame_bgr, max_dim):
    """Resize so the longest side is at most max_dim. Returns the
    frame unchanged if max_dim <= 0 or already small enough."""
    if max_dim <= 0:
        return frame_bgr
    H, W = frame_bgr.shape[:2]
    if max(H, W) <= max_dim:
        return frame_bgr
    scale = max_dim / max(H, W)
    return cv2.resize(frame_bgr, (int(W * scale), int(H * scale)))


def _bgr_to_rgba(frame_bgr):
    H, W = frame_bgr.shape[:2]
    rgba = np.empty((H, W, 4), dtype=np.uint8)
    rgba[..., 0] = frame_bgr[..., 2]
    rgba[..., 1] = frame_bgr[..., 1]
    rgba[..., 2] = frame_bgr[..., 0]
    rgba[..., 3] = 255
    return rgba


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_a", required=True,
                        help="Input mp4 for camera A (left).")
    parser.add_argument("--video_b", required=True,
                        help="Input mp4 for camera B (right).")
    parser.add_argument("--output", default="stitched_e2e.mp4",
                        help="Where to write the stitched mp4. "
                             "Default: stitched_e2e.mp4 in the cwd.")
    parser.add_argument("--n_frames", type=int, default=20,
                        help="How many paired frames to push through.")
    parser.add_argument("--max_resolution", type=int, default=0,
                        help="Resize inputs so the longest side is at "
                             "most this. 0 = no resize. Useful while "
                             "developing -- keeps each frame's bytes "
                             "well under default TCP buffers.")
    parser.add_argument("--no_fg", action="store_true",
                        help="Pass --no_fg through to the stitcher "
                             "(skips the depth-filtered FG mask, "
                             "which has long startup time).")
    parser.add_argument("--person_tracking", action="store_true",
                        help="Pass --person_tracking through to the "
                             "stitcher.")
    parser.add_argument("--python_exe", default=sys.executable,
                        help="Python interpreter to launch the "
                             "stitcher with. Default: this script's.")
    args = parser.parse_args()

    # --- Open listeners (Electron stand-in side) ----------------------
    control_listener = TCPListener("127.0.0.1", 0)
    frames_listener = TCPListener("127.0.0.1", 0)
    cp = control_listener.port
    fp = frames_listener.port
    print(f"[host] control port {cp}, frames port {fp}", flush=True)

    # --- Spawn the stitcher ------------------------------------------
    stitcher_cmd = [
        args.python_exe, "video_stitcher_seam_gpu.py",
        "--io", "pipe",
        "--control_port", str(cp),
        "--frames_port", str(fp),
        "--max_frames", str(args.n_frames),
    ]
    if args.no_fg:
        stitcher_cmd.append("--no_fg")
    if args.person_tracking:
        stitcher_cmd.append("--person_tracking")
    print(f"[host] spawning: {' '.join(stitcher_cmd)}", flush=True)
    proc = subprocess.Popen(stitcher_cmd, cwd=_REPO_ROOT)

    # --- Accept connections from the spawned stitcher ----------------
    print("[host] waiting for stitcher to connect...", flush=True)
    ctrl_t = control_listener.accept()
    frames_t = frames_listener.accept()
    print("[host] stitcher connected on both channels", flush=True)
    ctrl = ControlChannel(ctrl_t)

    # Shared state across threads.
    output_dims_event = threading.Event()
    output_dims = [None]      # (W, H)
    session_stopped_event = threading.Event()
    received = [0]
    writer_box = [None]
    error_messages = []

    # --- Control listener thread -------------------------------------
    def control_listener_thread():
        while True:
            msg = ctrl.recv()
            if msg is None:
                break
            t = msg.get("type")
            print(f"[host] control rx: {msg}", flush=True)
            if t == "session_started":
                output_dims[0] = (msg["output_width"],
                                  msg["output_height"])
                output_dims_event.set()
            elif t == "session_stopped":
                session_stopped_event.set()
                break
            elif t == "error":
                error_messages.append(msg)
                session_stopped_event.set()
                break

    # --- Frame receiver thread ---------------------------------------
    def frame_receiver_thread():
        try:
            for _ in range(args.n_frames):
                hdr_bytes = frames_t.read_exact(HEADER_SIZE)
                hdr = unpack_frame_header(hdr_bytes)
                if hdr["camera_index"] != OUTPUT_CAMERA_INDEX:
                    print(f"[host] WARN: unexpected camera_index "
                          f"{hdr['camera_index']} on output", flush=True)
                payload = frames_t.read_exact(hdr["payload_length"])
                rgba = np.frombuffer(payload, dtype=np.uint8).reshape(
                    hdr["height"], hdr["width"], 4,
                )
                bgr = np.ascontiguousarray(rgba[..., [2, 1, 0]])
                if writer_box[0] is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    wr = cv2.VideoWriter(
                        args.output, fourcc, 30.0,
                        (hdr["width"], hdr["height"]),
                    )
                    if not wr.isOpened():
                        raise RuntimeError(
                            f"failed to open output writer "
                            f"{args.output!r}"
                        )
                    writer_box[0] = wr
                    print(f"[host] first stitched frame "
                          f"{hdr['width']}x{hdr['height']}; opened "
                          f"writer {args.output}", flush=True)
                writer_box[0].write(bgr)
                received[0] += 1
        except ConnectionError:
            pass
        except Exception as e:
            print(f"[host] receiver error: {e}", flush=True)
            error_messages.append({"receiver_error": str(e)})

    # --- Read first frame from each input to learn input dimensions --
    cap_a = cv2.VideoCapture(args.video_a)
    cap_b = cv2.VideoCapture(args.video_b)
    if not (cap_a.isOpened() and cap_b.isOpened()):
        raise RuntimeError(
            f"could not open {args.video_a!r} or {args.video_b!r}"
        )
    ok_a, probe_a = cap_a.read()
    ok_b, probe_b = cap_b.read()
    if not (ok_a and ok_b):
        raise RuntimeError("could not read first frame from input videos")
    probe_a = _resize_long_side(probe_a, args.max_resolution)
    probe_b = _resize_long_side(probe_b, args.max_resolution)
    H_a, W_a = probe_a.shape[:2]
    H_b, W_b = probe_b.shape[:2]
    print(f"[host] input dims: cam_a={W_a}x{H_a}, cam_b={W_b}x{H_b}",
          flush=True)
    # Rewind so frame 0 is sent over the wire.
    cap_a.release(); cap_b.release()
    cap_a = cv2.VideoCapture(args.video_a)
    cap_b = cv2.VideoCapture(args.video_b)

    # --- Handshake ---------------------------------------------------
    ctrl.send(make_hello(versions=[PROTOCOL_VERSION]))
    ack = ctrl.recv()
    print(f"[host] hello_ack: {ack}", flush=True)
    if not (ack and ack.get("type") == "hello_ack"
            and ack.get("protocol_version") == PROTOCOL_VERSION):
        raise RuntimeError(f"bad hello_ack: {ack}")

    # --- start_session -----------------------------------------------
    ctrl.send(make_start_session(input_cameras=[
        {"index": 0, "label": "left",  "width": W_a, "height": H_a},
        {"index": 1, "label": "right", "width": W_b, "height": H_b},
    ]))
    print("[host] start_session sent", flush=True)

    # Async listener + receiver start now -- session_started arrives
    # after the stitcher has computed homography from frame 0 (we
    # have to start sending before then; spike caveat in protocol §11).
    ctrl_thread = threading.Thread(target=control_listener_thread,
                                   daemon=True)
    recv_thread = threading.Thread(target=frame_receiver_thread,
                                   daemon=True)
    ctrl_thread.start()
    recv_thread.start()

    # --- Send frames -------------------------------------------------
    print(f"[host] sending {args.n_frames} paired frames...", flush=True)
    base_ts = int(time.time() * 1e6)
    sent = 0
    for fidx in range(args.n_frames):
        ok_a, fa = cap_a.read()
        ok_b, fb = cap_b.read()
        if not (ok_a and ok_b):
            print(f"[host] input video EOF at frame {fidx}", flush=True)
            break
        fa = _resize_long_side(fa, args.max_resolution)
        fb = _resize_long_side(fb, args.max_resolution)
        ts = base_ts + fidx * 33_333  # ~30 fps spacing
        for cam_idx, frame_bgr in ((0, fa), (1, fb)):
            rgba = _bgr_to_rgba(frame_bgr)
            H, W = rgba.shape[:2]
            payload = rgba.tobytes()
            header = pack_frame_header(
                camera_index=cam_idx, timestamp_us=ts,
                width=W, height=H, payload_length=len(payload),
                fmt=FORMAT_RGBA8888,
            )
            frames_t.write(header + payload)
        sent += 1
        if fidx % 10 == 0 or fidx == args.n_frames - 1:
            print(f"[host]   sent frame {fidx}", flush=True)
    print(f"[host] all {sent} input frames sent", flush=True)

    # --- Wait for receiver to finish ---------------------------------
    recv_thread.join(timeout=60)
    print(f"[host] received {received[0]}/{args.n_frames} "
          f"stitched frames", flush=True)
    if writer_box[0] is not None:
        writer_box[0].release()
        print(f"[host] wrote {args.output}", flush=True)

    # --- Close frames channel; the stitcher will see EOF and exit ----
    try:
        frames_t.close()
    except Exception:
        pass

    # --- Wait for session_stopped on control --------------------------
    ctrl_thread.join(timeout=30)
    try:
        ctrl_t.close()
    except Exception:
        pass

    # --- Wait for child process --------------------------------------
    rc = proc.wait(timeout=30)
    print(f"[host] stitcher exit code: {rc}", flush=True)

    if error_messages:
        print(f"[host] FAIL: {error_messages}", flush=True)
        return 1
    if received[0] != sent:
        print(f"[host] FAIL: received {received[0]} / sent {sent}",
              flush=True)
        return 1
    if rc != 0:
        print(f"[host] FAIL: stitcher non-zero exit", flush=True)
        return 1
    print(f"[host] PASS  ({received[0]} frames round-tripped, "
          f"output written to {args.output})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
