"""
Smooth person-tracking crop for the stitched output.

When --person_tracking is enabled, the pipeline emits a 3:2 (W:H)
horizontal sub-crop of the autocropped panorama, centered on the
person closest to the cameras (largest blob in the canvas-space
person mask). The crop position is EMA-smoothed so the camera
appears to glide on a horizontal rail rather than snapping per
frame.

When no person is detected, the crop drifts back toward the centre
of the autocropped frame -- slower than the track-on-person rate
so a brief out-of-frame moment doesn't trigger a noticeable
re-centring.

The tracker takes ONLY the dilated warped person mask the pipeline
already produces for the seam-cost path (canvas-space, uint8 0/255).
We use cv2.connectedComponentsWithStats to pick the largest blob;
its centroid x is the target. No homography plumbing, no YOLO bbox
plumbing -- everything we need is already in that mask.

Cost: one cv2.connectedComponentsWithStats per frame on a uint8
mask. ~0.5 ms at typical canvas resolutions.
"""

import math

import cv2
import numpy as np


class PersonTracker:
    """
    Stateful EMA on the crop centre x-coordinate.

    update(mask, frame_w, frame_h) ingests one (canvas-space) person
    mask per frame; get_crop_bounds() returns the (x0, x1) bounds
    for the current frame's 3:2 horizontal crop.

    The EMA uses two time constants: a faster one when a person is
    in the mask (smooth_seconds, default 1.0 s), and a slower one
    when the frame is empty (drift_seconds, default 3.0 s) so the
    camera doesn't whip back to centre the instant someone steps
    out of frame.
    """

    def __init__(self, smooth_seconds=1.0, drift_seconds=3.0,
                 fps=25.0, aspect=1.5):
        if fps <= 0:
            raise ValueError("fps must be > 0")
        # alpha = 1 - exp(-1 / (tau * fps)) gives the per-frame EMA
        # weight equivalent to a continuous low-pass with time
        # constant tau seconds.
        self._alpha_track = 1.0 - math.exp(-1.0 / (smooth_seconds * fps))
        self._alpha_drift = 1.0 - math.exp(-1.0 / (drift_seconds * fps))
        self._aspect = float(aspect)

        # State, lazily initialised on first update.
        self._target_x = None     # smoothed crop-centre x in canvas px
        self._frame_w = None
        self._frame_h = None
        self._last_mode = None    # "track" or "drift" -- for debug logs

    @property
    def aspect(self):
        return self._aspect

    def update(self, person_mask_cpu, frame_w, frame_h):
        """
        Update the smoothed crop centre.

        Args:
            person_mask_cpu: (H, W) uint8 numpy array of the canvas-
                space person mask (0 / 255). Pass None or an empty
                mask to drift toward the centre.
            frame_w, frame_h: dimensions of the (autocropped) output
                frame. Constant across a run; passed every call
                because the tracker needs them to clamp / pick centre.
        """
        self._frame_w = frame_w
        self._frame_h = frame_h

        person_target = _largest_blob_centroid_x(person_mask_cpu)
        if person_target is None:
            # No person -> drift to centre.
            target = frame_w * 0.5
            alpha = self._alpha_drift
            self._last_mode = "drift"
        else:
            target = person_target
            alpha = self._alpha_track
            self._last_mode = "track"

        if self._target_x is None:
            # First update: snap to target so we don't slowly drift
            # from x=0 on the first frame.
            self._target_x = target
        else:
            self._target_x = (1.0 - alpha) * self._target_x + alpha * target

    def get_crop_bounds(self):
        """
        Returns (x0, x1) horizontal crop bounds (inclusive / exclusive),
        clamped to [0, frame_w]. Crop height is the full frame_h, so
        no y bounds. Returns None when update() has not been called
        yet (caller should treat as "no crop, pass full frame").
        """
        if self._target_x is None or self._frame_w is None:
            return None
        crop_w = int(round(self._frame_h * self._aspect))
        if crop_w >= self._frame_w:
            # Autocrop already narrower than 3:2 -- nothing to crop.
            return (0, self._frame_w)
        x0 = int(round(self._target_x - crop_w * 0.5))
        x0 = max(0, min(self._frame_w - crop_w, x0))
        return (x0, x0 + crop_w)

    def get_crop_size(self, frame_w, frame_h):
        """
        Return the (W, H) the writer should be initialised with --
        constant across the run since the aspect ratio is fixed.
        """
        crop_w = int(round(frame_h * self._aspect))
        crop_w = min(crop_w, frame_w)
        return (crop_w, frame_h)

    @property
    def last_mode(self):
        """'track' or 'drift' -- the last update()'s outcome. For logs."""
        return self._last_mode


def _largest_blob_centroid_x(person_mask_cpu):
    """
    Centroid x of the largest connected component in `person_mask_cpu`.
    Returns None if the mask is None / empty.

    The mask is uint8 0/255; treat any positive pixel as foreground.
    Using cv2.connectedComponentsWithStats which is very fast for
    sparse binary masks.
    """
    if person_mask_cpu is None:
        return None
    if not person_mask_cpu.any():
        return None
    binary = (person_mask_cpu > 0).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8,
    )
    if n <= 1:
        # Only the background label.
        return None
    # stats row 0 is background; skip it.
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return None
    largest_local = int(np.argmax(areas))
    return float(centroids[1 + largest_local, 0])
