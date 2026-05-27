"""
Frame I/O abstractions: FrameSource (input) and FrameSink (output).

The stitching pipeline consumes paired frames and emits stitched
frames. Today's file-mode operation reads two mp4 files and writes
one mp4 file; the upcoming live-mode operation reads paired RGBA
frames off a named pipe and writes the stitched output back over
the same pipe.

These ABCs let the pipeline body run unchanged in either mode --
it asks the source for "the next pair of input frames" and tells
the sink to "encode this stitched frame", without caring whether
the data came from disk or a peer process.

The file implementations (FileFrameSource / FileFrameSink) are
thin wrappers around the existing FrameSyncReader /
PrefetchingFrameReader / ThreadedVideoWriter so today's file-mode
CLI behaves identically. The pipe implementations live in
stitcher.pipe_io (PipeFrameSource / PipeFrameSink) and are wired
into the pipeline by the upcoming pipe-mode entry point.

Every frame carries a `timestamp_us` (microseconds) end-to-end.
File mode synthesises it from frame_idx / fps and ignores it on
write; pipe mode uses real capture timestamps from the Electron
host and puts them in the output header. Threading the value
through the pipeline now (even though file mode ignores it) means
the pipe-mode wiring is a small follow-up rather than another
cross-cutting change.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import cv2
import numpy as np

from stitcher.io_utils import PrefetchingFrameReader, ThreadedVideoWriter
from stitcher.sync_reader import FrameSyncReader


class FrameSource(ABC):
    """Source of paired BGR frames feeding the stitcher's compute loop."""

    @abstractmethod
    def open(self) -> None:
        """Initialise underlying resources. Call once before read_pair()."""

    @abstractmethod
    def read_pair(self) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
        """
        Return (frame_a_bgr, frame_b_bgr, timestamp_us) for the next
        paired frame, or None at end-of-stream / disconnect.
        """

    @abstractmethod
    def close(self) -> None:
        """Release underlying resources. Idempotent."""

    @property
    @abstractmethod
    def output_fps(self) -> float:
        """
        FPS used by the output writer + per-frame computations
        (PersonTracker EMA, FG recompute cadence). For file mode this
        comes from the input video's metadata; for pipe mode it's
        declared by the host in start_session.
        """

    def summary(self) -> str:
        """One-line startup summary. Default empty."""
        return ""

    def summary_post(self) -> str:
        """One-line shutdown summary. Default empty."""
        return ""


class FrameSink(ABC):
    """Sink for stitched BGR frames produced by the pipeline."""

    @abstractmethod
    def open(self, width: int, height: int, fps: float) -> None:
        """Open the sink for frames of the given size + framerate."""

    @abstractmethod
    def write(self, frame_bgr: np.ndarray, timestamp_us: int = 0) -> None:
        """Synchronous write of an already-materialized BGR frame."""

    @abstractmethod
    def write_async(self, pinned, event, post_sync_fn=None, free_cb=None,
                    timestamp_us: int = 0) -> None:
        """
        Async write of a GPU-pinned buffer: caller submits the buffer
        plus the cuda event that will signal when the GPU has finished
        filling it. The sink waits on the event, runs post_sync_fn(arr)
        on the resulting host view (debug overlays / tracking crop),
        writes the result, then calls free_cb() to release the pinned
        slot back to the caller's pool.
        """

    @abstractmethod
    def close(self) -> None:
        """Flush and release. Idempotent."""


# ---------------------------------------------------------------------------
# File-mode implementations
# ---------------------------------------------------------------------------


class FileFrameSource(FrameSource):
    """
    Reads paired frames from two video files via OpenCV. Composes the
    existing FrameSyncReader (FPS desync, frame pairing) and
    PrefetchingFrameReader (overlap decode with compute) -- no
    behavioural change from the pre-refactor pipeline; this class is
    a packaging step so the pipeline calls a single object.

    Frame 0 goes through the prefetcher like every other frame; the
    pipeline pulls it via read_pair() once for homography setup, then
    again for the main loop's first iteration.
    """

    def __init__(self, path_a: str, path_b: str, queue_depth: int = 3):
        self._path_a = path_a
        self._path_b = path_b
        self._queue_depth = queue_depth
        self._cap_a = None
        self._cap_b = None
        self._sync_reader = None
        self._prefetcher = None
        self._frame_idx = 0
        self._opened = False

    def open(self) -> None:
        self._cap_a = cv2.VideoCapture(self._path_a)
        self._cap_b = cv2.VideoCapture(self._path_b)
        if not self._cap_a.isOpened() or not self._cap_b.isOpened():
            raise RuntimeError(
                f"Could not open one of {self._path_a!r}, {self._path_b!r}"
            )
        fps_a = self._cap_a.get(cv2.CAP_PROP_FPS) or 25.0
        fps_b = self._cap_b.get(cv2.CAP_PROP_FPS) or 25.0
        self._sync_reader = FrameSyncReader(
            self._cap_a, self._cap_b, fps_a, fps_b,
        )
        self._prefetcher = PrefetchingFrameReader(
            self._sync_reader, queue_depth=self._queue_depth,
        )
        self._opened = True

    def read_pair(self) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
        if not self._opened:
            raise RuntimeError("FileFrameSource: read_pair() before open()")
        ok, fa, fb = self._prefetcher.read()
        if not ok:
            return None
        # Synthesize a monotonic timestamp from frame_idx + fps. The
        # pipeline doesn't use it for file mode, but the wire format
        # carries it through, so pipe-mode integration is a smaller
        # change later.
        ts_us = int(self._frame_idx * 1_000_000.0 / max(self.output_fps, 1e-6))
        self._frame_idx += 1
        return fa, fb, ts_us

    def close(self) -> None:
        if self._prefetcher is not None:
            self._prefetcher.close()
            self._prefetcher = None
        if self._cap_a is not None:
            self._cap_a.release()
            self._cap_a = None
        if self._cap_b is not None:
            self._cap_b.release()
            self._cap_b = None
        self._opened = False

    @property
    def output_fps(self) -> float:
        if self._sync_reader is None:
            return 25.0
        return self._sync_reader.output_fps

    def summary(self) -> str:
        return self._sync_reader.summary() if self._sync_reader else ""

    def summary_post(self) -> str:
        return self._sync_reader.summary_post() if self._sync_reader else ""


class FileFrameSink(FrameSink):
    """
    Writes stitched frames to an mp4 file via cv2.VideoWriter, wrapped
    in ThreadedVideoWriter so encoding doesn't stall the per-frame
    pipeline. Supports both sync (write) and GPU-pinned (write_async)
    handoffs. Timestamps are ignored -- the file container carries a
    constant fps from open().
    """

    def __init__(self, path: str, fourcc: str = "mp4v",
                 queue_depth: int = 4):
        self._path = path
        self._fourcc_code = fourcc
        self._queue_depth = queue_depth
        self._raw_writer = None
        self._threaded = None

    def open(self, width: int, height: int, fps: float) -> None:
        fourcc = cv2.VideoWriter_fourcc(*self._fourcc_code)
        self._raw_writer = cv2.VideoWriter(
            self._path, fourcc, fps, (int(width), int(height)),
        )
        if not self._raw_writer.isOpened():
            raise RuntimeError(
                f"Could not open output writer for {self._path!r}"
            )
        self._threaded = ThreadedVideoWriter(
            self._raw_writer, queue_depth=self._queue_depth,
        )

    def write(self, frame_bgr: np.ndarray, timestamp_us: int = 0) -> None:
        # File mode: constant-fps container, timestamp_us not used.
        self._threaded.write(frame_bgr)

    def write_async(self, pinned, event, post_sync_fn=None, free_cb=None,
                    timestamp_us: int = 0) -> None:
        # File mode: timestamp_us not used.
        self._threaded.write_async(pinned, event, post_sync_fn, free_cb)

    def close(self) -> None:
        if self._threaded is not None:
            self._threaded.close()
            self._threaded = None
        # ThreadedVideoWriter.close() also calls raw_writer.release().
        self._raw_writer = None
