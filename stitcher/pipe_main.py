"""
Pipe-mode entry point for VideoStitcher.

When the entry script is invoked with --io pipe, the Electron host
(or the test harness standing in for it) has already opened two TCP
listening sockets and spawned this process with --control-port and
--frames-port. This module:

  1. Connects to both ports as a client.
  2. Runs the protocol handshake (hello / hello_ack / start_session).
  3. Calls stitcher.pipeline.run() with a PipeFrameSource and a
     sink_factory that constructs a PipeFrameSink + emits
     session_started the moment the pipeline knows its output dims.
  4. On clean exit, sends session_stopped and closes both transports.

See docs/integration-protocol.md for the wire protocol. The session
lifecycle here matches §3.

Spike caveat (§12 open item): the host MAY start sending frames
immediately after start_session, before receiving session_started.
We currently compute the homography from frame 0 in the pipeline,
which means session_started can only be sent AFTER the first frame
has arrived and been processed. The host's first few frames sit in
the TCP buffer during that window; for the 320x240 RGBA test frames
in the harness this is well within typical Windows buffer limits.
Production will replace frame-0 homography with calibration loaded
from disk, and the host can then wait for session_started before
sending frames.
"""

from stitcher.pipe_io import (
    ControlChannel,
    PipeFrameSink,
    PipeFrameSource,
)
from stitcher.protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    make_error,
    make_hello_ack,
    make_session_started,
    make_session_stopped,
)
from stitcher.transport import TCPTransport


def run_pipe_session(args):
    """
    Pipe-mode dispatch from video_stitcher_seam_gpu.main(). Returns
    after the session ends cleanly; raises on protocol errors.

    Expects args to have:
        control_port: int
        frames_port: int
        (plus all the usual pipeline flags --yoloe_weights, etc.)
    """
    # Lazy import: pipeline.py pulls in YOLOE / heavy deps, and we'd
    # rather fail fast on bad args BEFORE loading those.
    from stitcher.pipeline import run as pipeline_run

    print(f"[pipe-main] connecting control 127.0.0.1:{args.control_port}",
          flush=True)
    ctrl_t = TCPTransport("127.0.0.1", args.control_port)
    print(f"[pipe-main] connecting frames  127.0.0.1:{args.frames_port}",
          flush=True)
    frames_t = TCPTransport("127.0.0.1", args.frames_port)

    ctrl = ControlChannel(ctrl_t)
    error_to_report = None
    try:
        # --- 1. Handshake --------------------------------------------
        hello = ctrl.recv()
        if hello is None or hello.get("type") != "hello":
            raise ProtocolError(f"expected hello, got {hello!r}")
        host_versions = hello.get("protocol_versions", [])
        if PROTOCOL_VERSION not in host_versions:
            raise ProtocolError(
                f"no shared protocol version: host advertises "
                f"{host_versions}, this stitcher speaks v{PROTOCOL_VERSION}"
            )
        ctrl.send(make_hello_ack(stitcher_version="0.0.0-spike"))
        print("[pipe-main] handshake OK", flush=True)

        # --- 2. start_session ----------------------------------------
        start = ctrl.recv()
        if start is None or start.get("type") != "start_session":
            raise ProtocolError(
                f"expected start_session, got {start!r}"
            )
        cams = start.get("input_cameras", [])
        print(f"[pipe-main] start_session: {len(cams)} input camera(s)",
              flush=True)
        # Calibration field is ignored in the spike; pipeline computes
        # homography from frame 0 like file mode.

        # --- 3. Run the pipeline with pipe-backed source / sink ------
        source = PipeFrameSource(frames_t, control_transport=ctrl_t)

        def sink_factory(w, h, fps):
            # First moment we know the stitched output size. Tell the
            # host so it can size its receive-side writer / canvas.
            ctrl.send(make_session_started(w, h))
            print(f"[pipe-main] session_started -> output {w}x{h} "
                  f"@ {fps:.2f} fps", flush=True)
            return PipeFrameSink(frames_t)

        try:
            pipeline_run(args, source=source, sink_factory=sink_factory)
        except Exception as e:
            # Capture so we can send a structured error message AFTER
            # the inner try/finally has had a chance to release
            # resources.
            error_to_report = e
            raise

        # --- 4. Clean session end ------------------------------------
        ctrl.send(make_session_stopped())
        print("[pipe-main] session_stopped sent", flush=True)
    finally:
        if error_to_report is not None:
            try:
                ctrl.send(make_error("pipeline_error",
                                     repr(error_to_report)))
            except ConnectionError:
                pass
        try:
            ctrl_t.close()
        except Exception:
            pass
        try:
            frames_t.close()
        except Exception:
            pass
