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


def compute_joint_gain_compensation_3cam(
    warped_L, warped_C, warped_R,
    overlap_LC_bbox, overlap_LC_in_bbox,
    overlap_CR_bbox, overlap_CR_in_bbox,
):
    """
    Per-channel BGR gain scalars (g_L, g_C, g_R) jointly solved across
    the two adjacent overlaps. Closed-form solution preserves overall
    brightness via the constraint g_L * g_C * g_R == 1.

    The pairwise function above can't be applied twice naively, because
    Center participates in both overlaps and the two pairwise solutions
    would disagree on Center's gain. We get a consistent triple instead.

    Math, per channel:
        Let alpha   = mean(L,  L<>C overlap)
            beta_L  = mean(C,  L<>C overlap)
            beta_R  = mean(C,  C<>R overlap)
            gamma   = mean(R,  C<>R overlap)

        We want corrected intensities to match in each overlap:
            alpha * g_L  =  beta_L * g_C            (1)
            beta_R * g_C  =  gamma * g_R            (2)
            g_L * g_C * g_R = 1                     (constraint)

        From (1):  g_L  = (beta_L / alpha) * g_C
        From (2):  g_R  = (beta_R / gamma) * g_C
        Substituting into the constraint:
            (beta_L / alpha) * g_C * g_C * (beta_R / gamma) * g_C  = 1
            g_C^3  =  (alpha * gamma) / (beta_L * beta_R)
            g_C    =  ((alpha * gamma) / (beta_L * beta_R)) ** (1/3)

        Then back-substitute g_L and g_R.

    Returns (g_L, g_C, g_R), each a (3,) float32 BGR vector. Means
    clipped to >= 1.0 before division to avoid division by ~0 in
    very dark images.
    """
    # L<>C overlap means
    xL0, yL0, xL1, yL1 = overlap_LC_bbox
    wL_in_LC = warped_L[yL0:yL1, xL0:xL1]
    wC_in_LC = warped_C[yL0:yL1, xL0:xL1]
    alpha = np.array(
        cv2.mean(wL_in_LC, mask=overlap_LC_in_bbox)[:3], dtype=np.float32
    )
    beta_L = np.array(
        cv2.mean(wC_in_LC, mask=overlap_LC_in_bbox)[:3], dtype=np.float32
    )

    # C<>R overlap means
    xR0, yR0, xR1, yR1 = overlap_CR_bbox
    wC_in_CR = warped_C[yR0:yR1, xR0:xR1]
    wR_in_CR = warped_R[yR0:yR1, xR0:xR1]
    beta_R = np.array(
        cv2.mean(wC_in_CR, mask=overlap_CR_in_bbox)[:3], dtype=np.float32
    )
    gamma = np.array(
        cv2.mean(wR_in_CR, mask=overlap_CR_in_bbox)[:3], dtype=np.float32
    )

    alpha  = np.clip(alpha,  1.0, None)
    beta_L = np.clip(beta_L, 1.0, None)
    beta_R = np.clip(beta_R, 1.0, None)
    gamma  = np.clip(gamma,  1.0, None)

    # Closed-form solve (per channel, vectorised across BGR).
    g_C_cubed = (alpha * gamma) / (beta_L * beta_R)
    g_C = np.cbrt(g_C_cubed).astype(np.float32)
    g_L = ((beta_L / alpha) * g_C).astype(np.float32)
    g_R = ((beta_R / gamma) * g_C).astype(np.float32)
    return g_L, g_C, g_R


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
    Two-frame warp. When the two source frames share a shape (the
    typical case), uploads A and B, stacks them into a (2, 3, H_src,
    W_src) tensor, and runs a SINGLE grid_sample against the
    precomputed (2, H_dst, W_dst, 2) grid stack — saves the
    kernel-launch + scheduling overhead vs two separate warp_gpu calls.

    When the two source frames have DIFFERENT shapes (e.g. cameras
    recording at different resolutions), falls back to two separate
    grid_sample calls — slightly slower but correct. The destination
    shape (H_dst, W_dst) is identical either way since both grids
    target the same output canvas.

    Returns (warped_a_t, warped_b_t), each (1, 3, H_dst, W_dst)
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
    if ta.shape[2:] == tb.shape[2:]:
        # Fast path: same source shape, single grid_sample on a batch
        # tensor of size 2.
        t = torch.cat([ta, tb], dim=0)
        warped = F.grid_sample(
            t, grid_pair_t,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        warped = warped.clamp(0, 255).to(torch.uint8)
        return warped[0:1], warped[1:2]
    # Fallback: different source shapes -> two grid_samples, one per
    # camera. We pull each camera's grid out of grid_pair_t along the
    # batch dim (grid_pair_t was built as cat([grid_a_t, grid_b_t], 0)).
    grid_a_t = grid_pair_t[0:1]
    grid_b_t = grid_pair_t[1:2]
    warped_a = F.grid_sample(
        ta, grid_a_t,
        mode="bilinear", padding_mode="zeros", align_corners=True,
    ).clamp(0, 255).to(torch.uint8)
    warped_b = F.grid_sample(
        tb, grid_b_t,
        mode="bilinear", padding_mode="zeros", align_corners=True,
    ).clamp(0, 255).to(torch.uint8)
    return warped_a, warped_b


def warp_triplet_gpu(
    frame_L_bgr_cpu, frame_C_bgr_cpu, frame_R_bgr_cpu,
    grid_triplet_t, device,
    gain_L_t=None, gain_C_t=None, gain_R_t=None,
    non_blocking=True,
):
    """
    Three-frame warp. Mirrors warp_pair_gpu's fast path: when all three
    source frames share a shape, uploads them, stacks into a
    (3, 3, H_src, W_src) tensor, and runs a single grid_sample against
    the precomputed (3, H_dst, W_dst, 2) grid stack -- one kernel launch
    for all three warps.

    Falls back to three separate grid_sample calls when the source
    shapes differ (mixed-resolution cameras). The destination shape is
    identical either way since all three grids target the same canvas.

    Returns (warped_L_t, warped_C_t, warped_R_t), each
    (1, 3, H_dst, W_dst) uint8 -- same per-camera shape warp_pair_gpu
    returns, so downstream per-frame code that only ever sees one
    camera's tensor at a time is unchanged.

    grid_triplet_t must be torch.cat([grid_L_t, grid_C_t, grid_R_t],
    dim=0). Build it once at startup; it's static.
    """
    tL = torch.from_numpy(frame_L_bgr_cpu).to(device, non_blocking=non_blocking)
    tC = torch.from_numpy(frame_C_bgr_cpu).to(device, non_blocking=non_blocking)
    tR = torch.from_numpy(frame_R_bgr_cpu).to(device, non_blocking=non_blocking)
    tL = tL.permute(2, 0, 1).unsqueeze(0).float()
    tC = tC.permute(2, 0, 1).unsqueeze(0).float()
    tR = tR.permute(2, 0, 1).unsqueeze(0).float()
    if gain_L_t is not None:
        tL = (tL * gain_L_t).clamp(0, 255)
    if gain_C_t is not None:
        tC = (tC * gain_C_t).clamp(0, 255)
    if gain_R_t is not None:
        tR = (tR * gain_R_t).clamp(0, 255)

    if tL.shape[2:] == tC.shape[2:] == tR.shape[2:]:
        # Fast path: batched grid_sample of size 3.
        t = torch.cat([tL, tC, tR], dim=0)
        warped = F.grid_sample(
            t, grid_triplet_t,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        warped = warped.clamp(0, 255).to(torch.uint8)
        return warped[0:1], warped[1:2], warped[2:3]

    # Fallback: different source shapes -> three separate grid_samples.
    grid_L_t = grid_triplet_t[0:1]
    grid_C_t = grid_triplet_t[1:2]
    grid_R_t = grid_triplet_t[2:3]
    warped_L = F.grid_sample(
        tL, grid_L_t, mode="bilinear",
        padding_mode="zeros", align_corners=True,
    ).clamp(0, 255).to(torch.uint8)
    warped_C = F.grid_sample(
        tC, grid_C_t, mode="bilinear",
        padding_mode="zeros", align_corners=True,
    ).clamp(0, 255).to(torch.uint8)
    warped_R = F.grid_sample(
        tR, grid_R_t, mode="bilinear",
        padding_mode="zeros", align_corners=True,
    ).clamp(0, 255).to(torch.uint8)
    return warped_L, warped_C, warped_R


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
