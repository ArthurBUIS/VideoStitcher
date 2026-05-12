"""
Motion detection via baseline subtraction.

For fixed cameras, the most robust "is something different from the
empty room" signal is just `|current - baseline| > threshold` applied
per camera, OR'd, dilated. Unlike MOG2 it doesn't fade stationary
objects into the background, and it reuses the already-warped frames
in the per-frame loop so the extra cost is small.

The motion mask is fed into the cost map as an additive penalty,
parallel to fg_mask, gated only by the person mask (so motion never
overrides person priority).

Baseline acquisition:
    * If both --motion_baseline_a and --motion_baseline_b are given,
      load the images from disk.
    * Otherwise, fall back to frame 0 of each input video. Convenient
      for quick tests when the room is empty at the very start of the
      recording.
"""

import cv2
import numpy as np
import torch

from stitcher.warp import dilate_gpu


def load_baseline_images(path_a, path_b):
    """Load per-camera baseline images from disk (BGR uint8)."""
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)
    if img_a is None:
        raise RuntimeError(f"Could not load motion baseline image: {path_a}")
    if img_b is None:
        raise RuntimeError(f"Could not load motion baseline image: {path_b}")
    return img_a, img_b


def grab_baseline_from_videos(video_a_path, video_b_path):
    """
    Open fresh VideoCapture objects, read frame 0 from each, and return
    the two frames. Used when --motion is enabled but no baseline image
    paths were provided.
    """
    cap_a = cv2.VideoCapture(video_a_path)
    cap_b = cv2.VideoCapture(video_b_path)
    try:
        ok_a, frame_a = cap_a.read()
        ok_b, frame_b = cap_b.read()
    finally:
        cap_a.release()
        cap_b.release()
    if not (ok_a and ok_b):
        raise RuntimeError("Could not read frame 0 from one of the input "
                           "videos as motion baseline fallback.")
    return frame_a, frame_b


def validate_baseline_shape(frame_a, frame_b, baseline_a, baseline_b):
    """Raise if a baseline image's shape doesn't match its camera's video."""
    if frame_a.shape != baseline_a.shape:
        raise RuntimeError(
            f"Motion baseline A shape {baseline_a.shape} does not match "
            f"camera A frame shape {frame_a.shape}. Re-record the baseline "
            f"at the same resolution as the video."
        )
    if frame_b.shape != baseline_b.shape:
        raise RuntimeError(
            f"Motion baseline B shape {baseline_b.shape} does not match "
            f"camera B frame shape {frame_b.shape}."
        )


def compute_motion_mask_gpu(warped_a_t, warped_b_t,
                            baseline_a_t, baseline_b_t,
                            threshold, dilate_radius,
                            overlap_bbox, overlap_in_bbox_t):
    """
    GPU motion mask. For each camera, compute the per-pixel sum-of-|BGR diff|
    against its baseline; OR the two binary masks; dilate; crop to bbox;
    AND with overlap shape.

    warped_*_t and baseline_*_t: (1, 3, H, W) uint8 tensors on the same device.
    threshold: scalar; sum-of-|BGR diff| above which a pixel is "moved".
    Returns: (H_bb, W_bb) uint8 tensor (0 or 255).
    """
    diff_a = (warped_a_t.float() - baseline_a_t.float()).abs().sum(dim=1)  # (1, H, W)
    diff_b = (warped_b_t.float() - baseline_b_t.float()).abs().sum(dim=1)
    motion = ((diff_a > threshold) | (diff_b > threshold))[0]  # (H, W) bool
    motion_u8 = motion.to(torch.uint8) * 255
    motion_u8 = dilate_gpu(motion_u8, dilate_radius)
    x0, y0, x1, y1 = overlap_bbox
    motion_bbox_t = motion_u8[y0:y1, x0:x1].contiguous()
    motion_bbox_t = torch.where(overlap_in_bbox_t > 0,
                                motion_bbox_t,
                                torch.zeros_like(motion_bbox_t))
    return motion_bbox_t


def compute_motion_mask_cpu(warped_a, warped_b,
                            baseline_a, baseline_b,
                            threshold, dilate_kernel,
                            overlap_bbox, overlap_in_bbox):
    """
    CPU motion mask, same logic via cv2.absdiff + numpy sum + threshold +
    cv2.dilate.

    Inputs: HxWx3 uint8 BGR numpy arrays (warped + gain-applied to match
    the current-frame pipeline).
    Returns: (H_bb, W_bb) uint8 numpy array (0 or 255).
    """
    diff_a = cv2.absdiff(warped_a, baseline_a)
    diff_b = cv2.absdiff(warped_b, baseline_b)
    total_a = diff_a.astype(np.int32).sum(axis=2)
    total_b = diff_b.astype(np.int32).sum(axis=2)
    motion = ((total_a > threshold) | (total_b > threshold)).astype(np.uint8) * 255
    if dilate_kernel is not None:
        motion = cv2.dilate(motion, dilate_kernel)
    x0, y0, x1, y1 = overlap_bbox
    motion_bbox = motion[y0:y1, x0:x1].copy()
    return cv2.bitwise_and(motion_bbox, overlap_in_bbox)
