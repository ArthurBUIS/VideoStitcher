"""I/O helpers: threaded video writer and debug overlay drawers."""

import queue
import threading

import cv2
import numpy as np


class PrefetchingFrameReader:
    """
    Wraps a FrameSyncReader (or any object with a `.read()` returning
    `(ok, frame_a, frame_b)`) and decodes the next paired frame on a
    background thread, so the pipeline's compute and the FFmpeg /
    OpenCV decode run in parallel.

    The decode side typically blocks the pipeline for ~10-20 ms per
    frame at 1080p; running it in parallel with compute hides that
    cost.

    Usage:
        prefetch = PrefetchingFrameReader(sync_reader, queue_depth=3)
        while True:
            ok, frame_a, frame_b = prefetch.read()
            if not ok:
                break
            ...
        prefetch.close()
    """

    _SENTINEL = object()

    def __init__(self, underlying_reader, queue_depth=3):
        self.reader = underlying_reader
        self.q = queue.Queue(maxsize=queue_depth)
        self._stopped = False
        self.exception = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self._stopped:
                ok, fa, fb = self.reader.read()
                if not ok:
                    break
                # Block until the consumer drains a slot, but check the
                # stop flag periodically so close() can unblock us.
                placed = False
                while not placed and not self._stopped:
                    try:
                        self.q.put((fa, fb), timeout=0.5)
                        placed = True
                    except queue.Full:
                        continue
        except Exception as e:
            self.exception = e
        finally:
            # Signal end-of-stream to the consumer (best-effort: skip
            # if the queue is full and we're stopping, since close()
            # will be draining anyway).
            try:
                self.q.put(self._SENTINEL, timeout=1.0)
            except queue.Full:
                pass

    def read(self):
        """Returns (ok, frame_a, frame_b). ok=False at end-of-stream."""
        if self.exception is not None:
            raise self.exception
        item = self.q.get()
        if item is self._SENTINEL:
            return False, None, None
        fa, fb = item
        return True, fa, fb

    def close(self):
        self._stopped = True
        # Drain the queue so the worker can unblock from a full-queue put.
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        self.thread.join(timeout=5)
        if self.exception is not None:
            raise self.exception


class _AsyncWriteItem:
    """Carries a pending GPU→CPU pinned buffer + cuda Event + post-sync
    transform to the writer thread. See ThreadedVideoWriter.write_async."""

    __slots__ = ("pinned", "event", "post_sync_fn", "free_cb")

    def __init__(self, pinned, event, post_sync_fn, free_cb):
        self.pinned = pinned
        self.event = event
        self.post_sync_fn = post_sync_fn
        self.free_cb = free_cb


class ThreadedVideoWriter:
    """
    Wrap cv2.VideoWriter on a background thread so encode/disk-write
    doesn't block the per-frame pipeline. Frames are copied into a
    bounded queue; the worker thread drains it.

    Supports two modes:
      - write(frame_bgr): synchronous handoff. The caller has already
        materialized the BGR frame on the host. The writer thread
        merely encodes.
      - write_async(pinned, event, post_sync_fn, free_cb): async handoff.
        The frame is still on the GPU at submission time; the writer
        thread waits on `event`, applies `post_sync_fn` to the pinned
        host view, encodes, then calls `free_cb` to release the pinned
        slot. This lets the composite worker move on to the next
        frame's kernel launches without waiting for the GPU.
    """

    _SENTINEL = object()

    def __init__(self, writer, queue_depth=4):
        self.writer = writer
        self.q = queue.Queue(maxsize=queue_depth)
        self.exception = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while True:
                item = self.q.get()
                if item is self._SENTINEL:
                    break
                if isinstance(item, _AsyncWriteItem):
                    # Wait for the GPU pyramid + pinned copy to finish.
                    item.event.synchronize()
                    arr = item.pinned.numpy()
                    if item.post_sync_fn is not None:
                        arr = item.post_sync_fn(arr)
                    self.writer.write(arr)
                    if item.free_cb is not None:
                        item.free_cb()
                else:
                    self.writer.write(item)
        except Exception as e:
            self.exception = e

    def write(self, frame_bgr):
        self.q.put(frame_bgr.copy())
        if self.exception is not None:
            raise self.exception

    def write_async(self, pinned, event, post_sync_fn=None, free_cb=None):
        self.q.put(_AsyncWriteItem(pinned, event, post_sync_fn, free_cb))
        if self.exception is not None:
            raise self.exception

    def close(self):
        self.q.put(self._SENTINEL)
        self.thread.join(timeout=30)
        if self.exception is not None:
            raise self.exception
        self.writer.release()


def draw_seam_overlay(canvas, seam_x_full, bbox):
    """Draw the DP seam as a red polyline on the canvas in place."""
    x0, y0, x1, y1 = bbox
    H_bb = y1 - y0
    ys = np.arange(H_bb) + y0
    xs = seam_x_full + x0
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    for i in range(len(pts) - 1):
        cv2.line(canvas, tuple(pts[i]), tuple(pts[i + 1]), (0, 0, 255), 2)


def draw_mask_overlay(canvas, mask_bbox, bbox, color=(0, 0, 255), alpha=0.35):
    """Composite a translucent colored overlay over the bbox region in place."""
    x0, y0, x1, y1 = bbox
    region = canvas[y0:y1, x0:x1]
    overlay = region.copy()
    overlay[mask_bbox > 0] = color
    cv2.addWeighted(overlay, alpha, region, 1 - alpha, 0, dst=region)
