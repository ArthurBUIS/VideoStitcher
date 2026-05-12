"""
Motion detection via baseline subtraction.

For fixed cameras, the "is something different from the empty room"
signal is built from a per-camera diff against a baseline frame,
OR'd, thresholded, dilated. Two diff strategies:

    * pixel : raw |current - baseline| on BGR. Cheap but sensitive
              to camera auto-exposure / auto-white-balance drift.
    * edges : |Sobel(current) - Sobel(baseline)| on grayscale.
              Robust to drift since edges depend on relative
              contrasts within a neighborhood, not absolute pixel
              values.

The motion mask is fed into the cost map as an additive penalty,
parallel to fg_mask, gated only by the person mask (so motion never
overrides person priority).

Baseline acquisition:
    * If both --motion_baseline_a and --motion_baseline_b are given,
      load the images from disk.
    * Otherwise, fall back to frame 0 of each input video.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from stitcher.warp import dilate_gpu


# Sobel kernels for the edge-based diff. L1 magnitude (|Gx| + |Gy|) is
# slightly cheaper than L2 and sufficient for the threshold-then-dilate
# pipeline that follows.
_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0],
                         [-2.0, 0.0, 2.0],
                         [-1.0, 0.0, 1.0]])
_SOBEL_Y = torch.tensor([[-1.0, -2.0, -1.0],
                         [ 0.0,  0.0,  0.0],
                         [ 1.0,  2.0,  1.0]])


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


# ---------------------------------------------------------------------------
# Edge-based diff (Sobel gradient magnitude)
# ---------------------------------------------------------------------------

def _bgr_to_gray_gpu(frame_t):
    """frame_t: (1, 3, H, W) uint8 BGR. Returns (1, 1, H, W) float gray
    using the standard BGR weights (0.114, 0.587, 0.299)."""
    return (0.114 * frame_t[:, 0:1].float()
            + 0.587 * frame_t[:, 1:2].float()
            + 0.299 * frame_t[:, 2:3].float())


def sobel_magnitude_gpu(frame_t):
    """
    Sobel L1 gradient magnitude (|Gx| + |Gy|) of a BGR frame on GPU.

    frame_t: (1, 3, H, W) uint8.
    Returns: (1, H, W) float32 tensor on the same device.
    """
    gray = _bgr_to_gray_gpu(frame_t)
    device = gray.device
    kx = _SOBEL_X.to(device).view(1, 1, 3, 3)
    ky = _SOBEL_Y.to(device).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return (gx.abs() + gy.abs())[0]  # (1, H, W)


def sobel_magnitude_cpu(frame_bgr):
    """Sobel L1 gradient magnitude on CPU. Returns (H, W) float32."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.abs(gx) + np.abs(gy)


def compute_motion_mask_gpu_edges(warped_a_t, warped_b_t,
                                  baseline_grad_a_t, baseline_grad_b_t,
                                  threshold, dilate_radius,
                                  overlap_bbox, overlap_in_bbox_t):
    """
    Edge-based motion mask (GPU). |Sobel(current) - Sobel(baseline)|
    thresholded per camera, OR'd, dilated, cropped to overlap bbox.

    Robust to brightness/color drift because edge magnitude only
    depends on relative contrasts within a 3x3 neighborhood.

    baseline_grad_*_t : precomputed (1, H, W) float Sobel magnitudes
                        of the baseline frames (see sobel_magnitude_gpu).
    """
    grad_a = sobel_magnitude_gpu(warped_a_t)
    grad_b = sobel_magnitude_gpu(warped_b_t)
    diff_a = (grad_a - baseline_grad_a_t).abs()
    diff_b = (grad_b - baseline_grad_b_t).abs()
    motion = ((diff_a > threshold) | (diff_b > threshold))[0]   # (H, W)
    motion_u8 = motion.to(torch.uint8) * 255
    motion_u8 = dilate_gpu(motion_u8, dilate_radius)
    x0, y0, x1, y1 = overlap_bbox
    motion_bbox_t = motion_u8[y0:y1, x0:x1].contiguous()
    motion_bbox_t = torch.where(overlap_in_bbox_t > 0,
                                motion_bbox_t,
                                torch.zeros_like(motion_bbox_t))
    return motion_bbox_t


def compute_motion_mask_cpu_edges(warped_a, warped_b,
                                  baseline_grad_a, baseline_grad_b,
                                  threshold, dilate_kernel,
                                  overlap_bbox, overlap_in_bbox):
    """CPU variant of compute_motion_mask_gpu_edges. baseline_grad_*
    are precomputed (H, W) float32 Sobel magnitudes."""
    grad_a = sobel_magnitude_cpu(warped_a)
    grad_b = sobel_magnitude_cpu(warped_b)
    diff_a = np.abs(grad_a - baseline_grad_a)
    diff_b = np.abs(grad_b - baseline_grad_b)
    motion = ((diff_a > threshold) | (diff_b > threshold)).astype(np.uint8) * 255
    if dilate_kernel is not None:
        motion = cv2.dilate(motion, dilate_kernel)
    x0, y0, x1, y1 = overlap_bbox
    motion_bbox = motion[y0:y1, x0:x1].copy()
    return cv2.bitwise_and(motion_bbox, overlap_in_bbox)


# ---------------------------------------------------------------------------
# Per-frame baseline renormalization (--motion_renorm)
# ---------------------------------------------------------------------------
#
# Cameras with auto-exposure drift the per-channel mean over time, so a
# pixel-diff against a static baseline can fire everywhere even when no
# real content changed. Renormalization rescales the current frame's
# per-channel mean (measured inside the overlap region) to match the
# baseline's, before the diff. Cancels global brightness/colour drift
# but not spatial lighting changes. Compose with any motion_method.

def compute_mean_in_overlap_gpu(warped_t, overlap_bbox, overlap_in_bbox_t):
    """warped_t: (1, 3, H, W) uint8. Returns (3,) float tensor of mean BGR
    over pixels where overlap_in_bbox_t > 0."""
    x0, y0, x1, y1 = overlap_bbox
    bb = warped_t[0, :, y0:y1, x0:x1].float()                # (3, H_bb, W_bb)
    mask = (overlap_in_bbox_t > 0).float()                   # (H_bb, W_bb)
    n = mask.sum().clamp(min=1.0)
    sums = (bb * mask.unsqueeze(0)).sum(dim=(1, 2))          # (3,)
    return sums / n


def compute_mean_in_overlap_cpu(warped, overlap_bbox, overlap_in_bbox):
    """CPU variant; uses cv2.mean for speed."""
    x0, y0, x1, y1 = overlap_bbox
    bb = warped[y0:y1, x0:x1]
    return np.array(cv2.mean(bb, mask=overlap_in_bbox)[:3], dtype=np.float32)


def renormalize_to_baseline_gpu(warped_t, baseline_mean_t,
                                overlap_bbox, overlap_in_bbox_t):
    """
    Per-channel rescale of warped_t so its mean over the overlap matches
    baseline_mean_t. Returns a new (1, 3, H, W) uint8 tensor.
    """
    current_mean = compute_mean_in_overlap_gpu(
        warped_t, overlap_bbox, overlap_in_bbox_t,
    )
    scale = (baseline_mean_t / current_mean.clamp(min=1.0)).view(1, 3, 1, 1)
    return (warped_t.float() * scale).clamp(0, 255).to(torch.uint8)


def renormalize_to_baseline_cpu(warped, baseline_mean,
                                overlap_bbox, overlap_in_bbox):
    """CPU variant of renormalize_to_baseline_gpu."""
    current_mean = compute_mean_in_overlap_cpu(
        warped, overlap_bbox, overlap_in_bbox,
    )
    scale = baseline_mean / np.maximum(current_mean, 1.0)
    return np.clip(warped.astype(np.float32) * scale,
                   0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Chrominance diff (LAB, drop the L channel)
# ---------------------------------------------------------------------------
#
# Convert to LAB, drop the L (lightness) channel, diff only on A and B
# (chrominance). Robust to brightness drift since brightness is in L.
# Weaker than the edge method for cameras with white-balance drift,
# since WB itself is a chrominance shift.

def _bgr_to_ab_gpu(frame_t):
    """
    BGR uint8 (1, 3, H, W) -> LAB A,B channels float (1, 2, H, W).

    No direct cv2-style LAB conversion in PyTorch, so we approximate via
    BGR -> sRGB -> XYZ -> LAB on the GPU. Uses the standard D65 white.
    Slightly heavier than the pixel diff but still cheap.
    """
    bgr = frame_t.float() / 255.0
    b = bgr[:, 0:1]
    g = bgr[:, 1:2]
    r = bgr[:, 2:3]

    # sRGB gamma (cheap linear approximation: skip the 2.4 power for
    # speed, since we only need to be consistent between current and
    # baseline, not perceptually accurate).
    rl, gl, bl = r, g, b

    # Linear RGB -> XYZ (sRGB / D65)
    X = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl
    Y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl
    Z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl

    # Normalize by D65 white
    X = X / 0.95047
    Z = Z / 1.08883

    # f(t) = t**(1/3) if t > eps else (kt + 16/116). Use a smooth-enough
    # numerical approximation via the standard formula:
    eps = 216.0 / 24389.0
    k = 24389.0 / 27.0

    def _f(t):
        return torch.where(t > eps,
                           t.clamp(min=1e-8).pow(1.0 / 3.0),
                           (k * t + 16.0) / 116.0)

    fy = _f(Y)
    A = 500.0 * (_f(X) - fy)
    B = 200.0 * (fy - _f(Z))

    return torch.cat([A, B], dim=1)  # (1, 2, H, W)


def _bgr_to_ab_cpu(frame_bgr):
    """CPU variant: cv2.cvtColor BGR->LAB, return only A, B channels."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab[:, :, 1:3]    # (H, W, 2)


def compute_motion_mask_gpu_chrominance(warped_a_t, warped_b_t,
                                        baseline_ab_a_t, baseline_ab_b_t,
                                        threshold, dilate_radius,
                                        overlap_bbox, overlap_in_bbox_t):
    """
    Chrominance-based motion mask (GPU). |LAB_AB(current) - LAB_AB(baseline)|
    summed across the two chroma channels, thresholded per camera, OR'd,
    dilated.

    baseline_ab_*_t : precomputed (1, 2, H, W) float A,B tensors.
    """
    ab_a = _bgr_to_ab_gpu(warped_a_t)            # (1, 2, H, W)
    ab_b = _bgr_to_ab_gpu(warped_b_t)
    diff_a = (ab_a - baseline_ab_a_t).abs().sum(dim=1)   # (1, H, W)
    diff_b = (ab_b - baseline_ab_b_t).abs().sum(dim=1)
    motion = ((diff_a > threshold) | (diff_b > threshold))[0]
    motion_u8 = motion.to(torch.uint8) * 255
    motion_u8 = dilate_gpu(motion_u8, dilate_radius)
    x0, y0, x1, y1 = overlap_bbox
    motion_bbox_t = motion_u8[y0:y1, x0:x1].contiguous()
    motion_bbox_t = torch.where(overlap_in_bbox_t > 0,
                                motion_bbox_t,
                                torch.zeros_like(motion_bbox_t))
    return motion_bbox_t


def compute_motion_mask_cpu_chrominance(warped_a, warped_b,
                                        baseline_ab_a, baseline_ab_b,
                                        threshold, dilate_kernel,
                                        overlap_bbox, overlap_in_bbox):
    """CPU variant of compute_motion_mask_gpu_chrominance."""
    ab_a = _bgr_to_ab_cpu(warped_a)
    ab_b = _bgr_to_ab_cpu(warped_b)
    diff_a = np.abs(ab_a - baseline_ab_a).sum(axis=2)
    diff_b = np.abs(ab_b - baseline_ab_b).sum(axis=2)
    motion = ((diff_a > threshold) | (diff_b > threshold)).astype(np.uint8) * 255
    if dilate_kernel is not None:
        motion = cv2.dilate(motion, dilate_kernel)
    x0, y0, x1, y1 = overlap_bbox
    motion_bbox = motion[y0:y1, x0:x1].copy()
    return cv2.bitwise_and(motion_bbox, overlap_in_bbox)


def precompute_baseline_ab_gpu(baseline_warped_t):
    return _bgr_to_ab_gpu(baseline_warped_t)


def precompute_baseline_ab_cpu(baseline_warped):
    return _bgr_to_ab_cpu(baseline_warped)
