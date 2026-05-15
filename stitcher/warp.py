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


def warp_pair_gpu(frame_a_bgr_cpu, frame_b_bgr_cpu,
                  grid_pair_t, device,
                  gain_a_t=None, gain_b_t=None,
                  non_blocking=True):
    """
    Batched two-frame warp. Uploads A and B, stacks them into a
    (2, 3, H_src, W_src) tensor, runs a SINGLE grid_sample against the
    precomputed (2, H_canvas, W_canvas, 2) grid stack. Saves the
    kernel-launch + scheduling overhead vs two separate warp_gpu calls.

    Returns (warped_a_t, warped_b_t), each (1, 3, H_canvas, W_canvas)
    uint8 — same shapes as warp_gpu returns, so downstream code is
    unchanged.
    """
    ta = torch.from_numpy(frame_a_bgr_cpu).to(device, non_blocking=non_blocking)
    tb = torch.from_numpy(frame_b_bgr_cpu).to(device, non_blocking=non_blocking)
    ta = ta.permute(2, 0, 1).unsqueeze(0).float()
    tb = tb.permute(2, 0, 1).unsqueeze(0).float()
    if gain_a_t is not None:
        ta = (ta * gain_a_t).clamp(0, 255)
    if gain_b_t is not None:
        tb = (tb * gain_b_t).clamp(0, 255)
    t = torch.cat([ta, tb], dim=0)
    warped = F.grid_sample(
        t, grid_pair_t,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    warped = warped.clamp(0, 255).to(torch.uint8)
    return warped[0:1], warped[1:2]


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
    """
    Binary dilation via max_pool2d. radius<=0 returns the mask unchanged.

    Square-kernel dilation is separable: dilating horizontally then
    vertically yields the same result as a single 2D pass, with work
    proportional to 2*k instead of k*k. For typical radius=10 (kernel
    21x21 = 441 vs 2*21 = 42 ops per pixel) that's a ~10x speed-up,
    which materially helps the motion mask path where the same dilate
    runs every frame on the bbox.
    """
    if radius <= 0:
        return mask_u8_t
    k = 2 * radius + 1
    m = mask_u8_t.float().unsqueeze(0).unsqueeze(0)
    m = F.max_pool2d(m, kernel_size=(1, k), stride=1, padding=(0, radius))
    m = F.max_pool2d(m, kernel_size=(k, 1), stride=1, padding=(radius, 0))
    return m[0, 0].to(torch.uint8)
