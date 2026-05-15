"""
Motion detection via baseline subtraction.

For fixed cameras, the "is something different from the empty room"
signal is built from a per-camera diff against a baseline frame,
OR'd, thresholded, dilated. Three diff strategies:

    * pixel       : raw |current - baseline| on BGR. Cheap but sensitive
                    to camera auto-exposure / auto-white-balance drift.
    * edges       : |Sobel(current) - Sobel(baseline)| on grayscale.
                    Robust to drift since edges depend on relative
                    contrasts within a 3x3 neighborhood, not absolute
                    pixel values.
    * chrominance : |LAB_AB(current) - LAB_AB(baseline)|, dropping the
                    L (lightness) channel. Robust to brightness drift,
                    not to true white-balance drift.

The motion mask is fed into the cost map as an additive penalty,
parallel to fg_mask, gated only by the person mask (so motion never
overrides person priority).

Performance note: every per-frame helper here operates on
bbox-cropped tensors (the overlap bbox is the only region whose
motion mask the seam actually consumes). Baselines + baseline
gradients + baseline LAB-AB are pre-cropped to the bbox once at
startup; per frame the warped tensors are sliced to the bbox by the
caller (pipeline.compute_one) before being handed to any of these
functions. On the GPU side this cuts the diff/threshold/dilate work
from `output_H x output_W` down to `H_bbox x W_bbox`.

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


# ---------------------------------------------------------------------------
# Bbox slicing helpers (pre-crop baselines once at startup; pipeline
# slices warped frames per call before invoking the mask helpers below).
# ---------------------------------------------------------------------------

def crop_to_bbox_gpu(warped_t, overlap_bbox):
    """warped_t: (1, 3, H, W). Returns (3, H_bb, W_bb) contiguous view-copy
    over the overlap bbox. Channel-first because that's what the diff
    helpers expect."""
    x0, y0, x1, y1 = overlap_bbox
    return warped_t[0, :, y0:y1, x0:x1].contiguous()


def crop_to_bbox_cpu(warped, overlap_bbox):
    """warped: (H, W, 3). Returns the bbox slice (no copy)."""
    x0, y0, x1, y1 = overlap_bbox
    return warped[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Half-resolution helpers (motion mask runs at half the bbox res — see
# pipeline.motion_worker). All per-frame diff / threshold / dilate work
# moves from (H_bb, W_bb) to (H_bb/2, W_bb/2), which is ~4x cheaper.
# Final motion mask is nearest-upsampled back to full bbox before the
# AND with overlap_in_bbox_t. Block granularity in the cost map is 2px,
# easily absorbed by motion_dilate's halo and seam_lambda's smoothing.
# ---------------------------------------------------------------------------

MOTION_DOWNSCALE = 2


def downsample_image_half_gpu(image_bb_t):
    """image_bb_t: (3, H_bb, W_bb) uint8 BGR. Returns (3, H/2, W/2)
    uint8 via 2x2 avg_pool (box downsample)."""
    out = F.avg_pool2d(
        image_bb_t.float().unsqueeze(0),
        kernel_size=MOTION_DOWNSCALE, stride=MOTION_DOWNSCALE,
    )
    return out[0].to(torch.uint8)


def downsample_mask_half_gpu(mask_bb_t):
    """mask_bb_t: (H_bb, W_bb) uint8 binary. Returns (H/2, W/2) uint8
    via max_pool — conservative (any-set-in-2x2-block sets output)."""
    out = F.max_pool2d(
        mask_bb_t.float().unsqueeze(0).unsqueeze(0),
        kernel_size=MOTION_DOWNSCALE, stride=MOTION_DOWNSCALE,
    )
    return out[0, 0].to(torch.uint8)


def upsample_mask_to_bbox_gpu(mask_half_t, target_hw):
    """Nearest-neighbor upsample of a half-res binary mask back to full
    bbox. Returns (H_bb, W_bb) uint8."""
    out = F.interpolate(
        mask_half_t.unsqueeze(0).unsqueeze(0).float(),
        size=target_hw, mode="nearest",
    )
    return out[0, 0].to(torch.uint8)


def downsample_image_half_cpu(image_bb):
    """image_bb: (H_bb, W_bb, 3) uint8. Returns (H/2, W/2, 3)."""
    return cv2.resize(
        image_bb,
        (image_bb.shape[1] // MOTION_DOWNSCALE,
         image_bb.shape[0] // MOTION_DOWNSCALE),
        interpolation=cv2.INTER_AREA,
    )


def downsample_mask_half_cpu(mask_bb):
    """mask_bb: (H_bb, W_bb) uint8 binary. Returns (H/2, W/2)."""
    # cv2 doesn't have a max-pool resize; INTER_NEAREST + dilate before
    # would be exact but unnecessarily expensive. Use a max-pool via
    # numpy: reshape into 2x2 blocks, take the max.
    h, w = mask_bb.shape
    h2 = (h // MOTION_DOWNSCALE) * MOTION_DOWNSCALE
    w2 = (w // MOTION_DOWNSCALE) * MOTION_DOWNSCALE
    mb = mask_bb[:h2, :w2]
    return mb.reshape(h2 // MOTION_DOWNSCALE, MOTION_DOWNSCALE,
                      w2 // MOTION_DOWNSCALE, MOTION_DOWNSCALE) \
             .max(axis=(1, 3))


def upsample_mask_to_bbox_cpu(mask_half, target_hw):
    """Nearest upsample of half-res mask back to full bbox."""
    th, tw = target_hw
    return cv2.resize(mask_half, (tw, th), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------
# Pixel-diff motion mask
# ---------------------------------------------------------------------------

def compute_motion_mask_gpu(wa_bb_t, wb_bb_t,
                            ba_bb_t, bb_bb_t,
                            threshold, dilate_radius,
                            overlap_in_bbox_t):
    """
    GPU motion mask on bbox-cropped tensors. For each camera, compute
    the per-pixel sum-of-|BGR diff| against its baseline; OR the two
    binary masks; dilate; AND with the overlap shape.

    Inputs: (3, H_bb, W_bb) uint8 tensors for the current warped + the
    baseline-warped frames, all already cropped to the overlap bbox.
    Returns: (H_bb, W_bb) uint8 tensor (0 or 255).
    """
    diff_a = (wa_bb_t.float() - ba_bb_t.float()).abs().sum(dim=0)   # (H, W)
    diff_b = (wb_bb_t.float() - bb_bb_t.float()).abs().sum(dim=0)
    motion = (diff_a > threshold) | (diff_b > threshold)
    motion_u8 = motion.to(torch.uint8) * 255
    motion_u8 = dilate_gpu(motion_u8, dilate_radius)
    motion_u8 = torch.where(overlap_in_bbox_t > 0,
                            motion_u8,
                            torch.zeros_like(motion_u8))
    return motion_u8


def compute_motion_mask_cpu(wa_bb, wb_bb,
                            ba_bb, bb_bb,
                            threshold, dilate_kernel,
                            overlap_in_bbox):
    """
    CPU motion mask on bbox-cropped frames, same logic as the GPU path
    via cv2.absdiff + numpy + cv2.dilate.
    """
    diff_a = cv2.absdiff(wa_bb, ba_bb)
    diff_b = cv2.absdiff(wb_bb, bb_bb)
    total_a = diff_a.astype(np.int32).sum(axis=2)
    total_b = diff_b.astype(np.int32).sum(axis=2)
    motion = ((total_a > threshold) | (total_b > threshold)).astype(np.uint8) * 255
    if dilate_kernel is not None:
        motion = cv2.dilate(motion, dilate_kernel)
    return cv2.bitwise_and(motion, overlap_in_bbox)


# ---------------------------------------------------------------------------
# Edge-based diff (Sobel gradient magnitude)
# ---------------------------------------------------------------------------

def _bgr_to_gray_gpu_bb(frame_bb_t):
    """frame_bb_t: (3, H, W) uint8 BGR. Returns (1, 1, H, W) float gray."""
    return (0.114 * frame_bb_t[0:1].float()
            + 0.587 * frame_bb_t[1:2].float()
            + 0.299 * frame_bb_t[2:3].float()).unsqueeze(0)


def sobel_magnitude_gpu_bb(frame_bb_t):
    """
    Sobel L1 gradient magnitude (|Gx| + |Gy|) on a bbox-cropped BGR
    frame on GPU.

    frame_bb_t: (3, H, W) uint8.
    Returns: (H, W) float32 tensor.
    """
    gray = _bgr_to_gray_gpu_bb(frame_bb_t)
    device = gray.device
    kx = _SOBEL_X.to(device).view(1, 1, 3, 3)
    ky = _SOBEL_Y.to(device).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return (gx.abs() + gy.abs())[0, 0]   # (H, W)


def sobel_magnitude_cpu_bb(frame_bb):
    """Sobel L1 gradient magnitude on a bbox-cropped CPU BGR frame.
    Returns (H, W) float32."""
    gray = cv2.cvtColor(frame_bb, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.abs(gx) + np.abs(gy)


def compute_motion_mask_gpu_edges(wa_bb_t, wb_bb_t,
                                  baseline_grad_a_bb_t, baseline_grad_b_bb_t,
                                  threshold, dilate_radius,
                                  overlap_in_bbox_t):
    """
    Edge-based motion mask (GPU) on bbox-cropped tensors.
    |Sobel(current) - Sobel(baseline)| thresholded per camera, OR'd,
    dilated, AND'd with the overlap shape.

    baseline_grad_*_bb_t : precomputed (H_bb, W_bb) Sobel magnitudes of
                          the bbox-cropped baseline frames (see
                          sobel_magnitude_gpu_bb).
    """
    grad_a = sobel_magnitude_gpu_bb(wa_bb_t)
    grad_b = sobel_magnitude_gpu_bb(wb_bb_t)
    diff_a = (grad_a - baseline_grad_a_bb_t).abs()
    diff_b = (grad_b - baseline_grad_b_bb_t).abs()
    motion = (diff_a > threshold) | (diff_b > threshold)
    motion_u8 = motion.to(torch.uint8) * 255
    motion_u8 = dilate_gpu(motion_u8, dilate_radius)
    motion_u8 = torch.where(overlap_in_bbox_t > 0,
                            motion_u8,
                            torch.zeros_like(motion_u8))
    return motion_u8


def compute_motion_mask_cpu_edges(wa_bb, wb_bb,
                                  baseline_grad_a_bb, baseline_grad_b_bb,
                                  threshold, dilate_kernel,
                                  overlap_in_bbox):
    """CPU variant of compute_motion_mask_gpu_edges, bbox-cropped."""
    grad_a = sobel_magnitude_cpu_bb(wa_bb)
    grad_b = sobel_magnitude_cpu_bb(wb_bb)
    diff_a = np.abs(grad_a - baseline_grad_a_bb)
    diff_b = np.abs(grad_b - baseline_grad_b_bb)
    motion = ((diff_a > threshold) | (diff_b > threshold)).astype(np.uint8) * 255
    if dilate_kernel is not None:
        motion = cv2.dilate(motion, dilate_kernel)
    return cv2.bitwise_and(motion, overlap_in_bbox)


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

def compute_mean_in_overlap_gpu(warped_bb_t, overlap_in_bbox_t):
    """warped_bb_t: (3, H_bb, W_bb) uint8. Returns (3,) float tensor of
    mean BGR over pixels where overlap_in_bbox_t > 0."""
    bb = warped_bb_t.float()                                 # (3, H_bb, W_bb)
    mask = (overlap_in_bbox_t > 0).float()                   # (H_bb, W_bb)
    n = mask.sum().clamp(min=1.0)
    sums = (bb * mask.unsqueeze(0)).sum(dim=(1, 2))          # (3,)
    return sums / n


def compute_mean_in_overlap_cpu(warped_bb, overlap_in_bbox):
    """CPU variant; uses cv2.mean for speed."""
    return np.array(cv2.mean(warped_bb, mask=overlap_in_bbox)[:3],
                    dtype=np.float32)


def renormalize_to_baseline_gpu(warped_bb_t, baseline_mean_t,
                                overlap_in_bbox_t):
    """
    Per-channel rescale of the bbox-cropped warped frame so its mean
    over the overlap matches baseline_mean_t. Returns a new
    (3, H_bb, W_bb) uint8 tensor.
    """
    current_mean = compute_mean_in_overlap_gpu(
        warped_bb_t, overlap_in_bbox_t,
    )
    scale = (baseline_mean_t / current_mean.clamp(min=1.0)).view(3, 1, 1)
    return (warped_bb_t.float() * scale).clamp(0, 255).to(torch.uint8)


def renormalize_to_baseline_cpu(warped_bb, baseline_mean, overlap_in_bbox):
    """CPU variant of renormalize_to_baseline_gpu."""
    current_mean = compute_mean_in_overlap_cpu(warped_bb, overlap_in_bbox)
    scale = baseline_mean / np.maximum(current_mean, 1.0)
    return np.clip(warped_bb.astype(np.float32) * scale,
                   0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Chrominance diff (LAB, drop the L channel)
# ---------------------------------------------------------------------------
#
# Convert to LAB, drop the L (lightness) channel, diff only on A and B
# (chrominance). Robust to brightness drift since brightness is in L.
# Weaker than the edge method for cameras with white-balance drift,
# since WB itself is a chrominance shift.

def _bgr_to_ab_gpu_bb(frame_bb_t):
    """
    BGR uint8 (3, H, W) -> LAB A,B channels float (2, H, W).

    No direct cv2-style LAB conversion in PyTorch, so we approximate via
    BGR -> sRGB -> XYZ -> LAB on the GPU. Uses the standard D65 white.
    Slightly heavier than the pixel diff but still cheap.
    """
    bgr = frame_bb_t.float() / 255.0
    b = bgr[0:1]
    g = bgr[1:2]
    r = bgr[2:3]

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

    return torch.cat([A, B], dim=0)   # (2, H, W)


def _bgr_to_ab_cpu_bb(frame_bb):
    """CPU variant: cv2.cvtColor BGR->LAB, return only A, B channels."""
    lab = cv2.cvtColor(frame_bb, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab[:, :, 1:3]


def compute_motion_mask_gpu_chrominance(wa_bb_t, wb_bb_t,
                                        baseline_ab_a_bb_t, baseline_ab_b_bb_t,
                                        threshold, dilate_radius,
                                        overlap_in_bbox_t):
    """
    Chrominance-based motion mask (GPU) on bbox-cropped tensors.

    baseline_ab_*_bb_t : precomputed (2, H_bb, W_bb) A,B tensors.
    """
    ab_a = _bgr_to_ab_gpu_bb(wa_bb_t)
    ab_b = _bgr_to_ab_gpu_bb(wb_bb_t)
    diff_a = (ab_a - baseline_ab_a_bb_t).abs().sum(dim=0)   # (H, W)
    diff_b = (ab_b - baseline_ab_b_bb_t).abs().sum(dim=0)
    motion = (diff_a > threshold) | (diff_b > threshold)
    motion_u8 = motion.to(torch.uint8) * 255
    motion_u8 = dilate_gpu(motion_u8, dilate_radius)
    motion_u8 = torch.where(overlap_in_bbox_t > 0,
                            motion_u8,
                            torch.zeros_like(motion_u8))
    return motion_u8


def compute_motion_mask_cpu_chrominance(wa_bb, wb_bb,
                                        baseline_ab_a_bb, baseline_ab_b_bb,
                                        threshold, dilate_kernel,
                                        overlap_in_bbox):
    """CPU variant of compute_motion_mask_gpu_chrominance."""
    ab_a = _bgr_to_ab_cpu_bb(wa_bb)
    ab_b = _bgr_to_ab_cpu_bb(wb_bb)
    diff_a = np.abs(ab_a - baseline_ab_a_bb).sum(axis=2)
    diff_b = np.abs(ab_b - baseline_ab_b_bb).sum(axis=2)
    motion = ((diff_a > threshold) | (diff_b > threshold)).astype(np.uint8) * 255
    if dilate_kernel is not None:
        motion = cv2.dilate(motion, dilate_kernel)
    return cv2.bitwise_and(motion, overlap_in_bbox)


def precompute_baseline_ab_gpu(baseline_bb_t):
    """baseline_bb_t: (3, H_bb, W_bb) uint8 -> (2, H_bb, W_bb) float."""
    return _bgr_to_ab_gpu_bb(baseline_bb_t)


def precompute_baseline_ab_cpu(baseline_bb):
    return _bgr_to_ab_cpu_bb(baseline_bb)
