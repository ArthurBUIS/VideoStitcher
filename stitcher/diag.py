"""Lightweight diagnostic logger for the stitcher pipeline.

Writes timestamped lines to a file (line-buffered, so `tail -f` shows
output in real time) or, when no path is given, to stdout. Thread-
safe: worker threads call write() without coordination.

Wired together with the --profile / --diag_log_file CLI flags:
    --profile alone              -> rolling profile lines to stdout
                                    (legacy behaviour)
    --profile + --diag_log_file  -> rolling profile + queue-depth
                                    samples + bottleneck hints to
                                    the file; stdout stays clean
                                    for the host-bridge to consume.

The intent is that the diagnostic file is the place where pipeline
internals (per-stage StageTimer summaries, queue depths, per-frame
waits, etc.) get printed, while stdout keeps showing the short
operator-visible status (`[info] N-cam: frame N  X fps over ...`).
That separation lets the host UI log keep its signal-to-noise
ratio while the operator still has a deeper view available on
disk when they need to debug.
"""

import threading
import time


class DiagLogger:
    """Thread-safe diagnostic line writer.

    Usage:
        diag = DiagLogger(path="C:/.../stitcher-diag/123.log")
        diag.write("session started")
        diag.section("rolling profile  queues: compute_q=2 composite_q=1")
        diag.kv(2, "compute", "avg=120ms")
        ...
        diag.close()

    Lifecycle:
        - Pass None or omit `path` for stdout fallback (legacy mode).
        - close() flushes and shuts the file handle; safe to call
          multiple times.
    """

    def __init__(self, path=None):
        self._lock = threading.Lock()
        self._fh = None
        self._path = path
        if path:
            # Line-buffered so tail -f works. Truncate per-run
            # (one file per Python sidecar lifetime, host picks a
            # fresh filename each spawn).
            self._fh = open(path, "w", buffering=1, encoding="utf-8")

    @property
    def path(self):
        return self._path

    def is_file(self):
        return self._fh is not None

    def write(self, line):
        ts = time.strftime("%H:%M:%S")
        out = f"[{ts}] {line}"
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.write(out + "\n")
                except Exception:
                    # Disk full / handle gone -- silently degrade
                    # rather than crash the pipeline.
                    pass
            else:
                # Stdout fallback: rely on print's own line buffering.
                print(out, flush=True)

    def section(self, header):
        """Print a blank line + a `=== header ===` line. Used to
        visually break up rolling profile blocks."""
        self.write("")
        self.write(f"=== {header} ===")

    def kv(self, indent, name, value):
        """Aligned `key                value` line. The width keeps
        StageTimer summaries lined up across stages."""
        pad = " " * indent
        self.write(f"{pad}{name:<22s} {value}")

    def close(self):
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.flush()
                    self._fh.close()
                except Exception:
                    pass
                self._fh = None
