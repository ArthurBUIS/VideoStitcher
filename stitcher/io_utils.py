"""I/O helpers: threaded video writer and debug overlay drawers."""

import queue
import threading

import cv2
import numpy as np


class ThreadedVideoWriter:
    """
    Wrap cv2.VideoWriter on a background thread so encode/disk-write
    doesn't block the per-frame pipeline. Frames are copied into a
    bounded queue; the worker thread drains it.
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
                self.writer.write(item)
        except Exception as e:
            self.exception = e

    def write(self, frame_bgr):
        self.q.put(frame_bgr.copy())
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
