"""
Per-frame image warping + gain compensation.

GPU path uses PyTorch grid_sample on a precomputed sampling tensor;
CPU path uses cv2.LUT for gain + cv2.remap (called from the pipeline).
Mask warping + dilation use the same grid_sample tensor with nearest
interpolation + max_pool2d.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gain compensation
# ---------------------------------------------------------------------------

def compute_gain_compensation(warped_a, warped_b, overlap_bbox, overlap_in_bbox):
    """Per-channel BGR gain scalars to match mean exposure in the overlap."""
    x0, y0, x1, y1 = overlap_bbox
    wa = warped_a[y0:y1, x0:x1]
    wb = warped_b[y0:y1, x0:x1]
    mean_a = np.array(cv2.mean(wa, mask=overlap_in_bbox)[:3], dtype=np.float32)
    mean_b = np.array(cv2.mean(wb, mask=overlap_in_bbox)[:3], dtype=np.float32)
    mean_a = np.clip(mean_a, 1.0, None)
    mean_b = np.clip(mean_b, 1.0, None)
    target = np.sqrt(mean_a * mean_b)
    g_a = (target / mean_a).astype(np.float32)
    g_b = (target / mean_b).astype(np.float32)
    return g_a, g_b


def build_gain_lut(gains_bgr):
    """CPU: build a (1, 256, 3) lookup table for cv2.LUT."""
    x = np.arange(256, dtype=np.float32)
    scaled = x[:, None] * gains_bgr[None, :]
    scaled = np.clip(scaled, 0, 255).astype(np.uint8)
    return scaled.reshape(1, 256, 3)


def apply_gain_lut(img_uint8, lut):
    """CPU: apply a gain LUT to a BGR frame."""
    return cv2.LUT(img_uint8, lut)


def build_gain_tensor(gains_bgr, device):
    """GPU: build a (1, 3, 1, 1) gain tensor (BGR), folded into the warp."""
    t = torch.from_numpy(gains_bgr).to(device).view(1, 3, 1, 1)
    return t


# ---------------------------------------------------------------------------
# GPU warp
# ---------------------------------------------------------------------------

def build_grid_sample_tensor(map_x, map_y, src_shape, device):
    """
    Convert cv2.remap-style map_x/map_y (in source pixel coords) into a
    grid_sample tensor (in normalized [-1, 1] coords, shape [1, H, W, 2]).
    Used to drive F.grid_sample for both frame and mask warps.
    """
    H_src, W_src = src_shape[:2]
    grid_x = 2.0 * map_x / max(W_src - 1, 1) - 1.0
    grid_y = 2.0 * map_y / max(H_src - 1, 1) - 1.0
    grid_np = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
    grid_t = torch.from_numpy(grid_np).unsqueeze(0).to(device)
    return grid_t


def warp_gpu(frame_bgr_cpu, grid_t, device, gain_t=None, non_blocking=True):
    """
    Upload a BGR frame to GPU and warp it to canvas via grid_sample.
    Optional gain_t (BGR scalars) is applied multiplicatively during the
    upload (saves a separate pass).
    """
    t = torch.from_numpy(frame_bgr_cpu).to(device, non_blocking=non_blocking)
    t = t.permute(2, 0, 1).unsqueeze(0).float()
    if gain_t is not None:
        t = t * gain_t
        t = t.clamp(0, 255)
    warped = F.grid_sample(
        t, grid_t,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    warped = warped.clamp(0, 255).to(torch.uint8)
    return warped


# ---------------------------------------------------------------------------
# GPU mask warp + dilation
# ---------------------------------------------------------------------------

def warp_mask_gpu(mask_t, grid_t):
    """Warp a uint8 mask (H, W) to canvas via grid_sample (nearest interp)."""
    m = mask_t.float().unsqueeze(0).unsqueeze(0)
    if m.max() > 1.5:
        m = m / 255.0
    warped = F.grid_sample(
        m, grid_t,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )
    out = (warped[0, 0] * 255).clamp(0, 255).to(torch.uint8)
    return out


def dilate_gpu(mask_u8_t, radius):
    """Binary dilation via max_pool2d. radius<=0 returns the mask unchanged."""
    if radius <= 0:
        return mask_u8_t
    k = 2 * radius + 1
    m = mask_u8_t.float().unsqueeze(0).unsqueeze(0)
    dilated = F.max_pool2d(m, kernel_size=k, stride=1, padding=radius)
    return dilated[0, 0].to(torch.uint8)
