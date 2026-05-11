"""
Seam cost map + dynamic-programming seam finder.

The cost map is built from photometric squared-BGR difference inside
the overlap bbox, smoothed across frames via an EMA, then has additive
penalties injected for pixels that fall on the person mask, the static
FG mask, and the left/right "edge margin" band. The DP seam is found
on a downscaled version of that cost; a quadratic regularizer keeps
the seam close to the previous frame's seam for stability.
"""

import cv2
import numpy as np
import torch


# Default penalty amplitudes (used as argparse defaults from cli.py).
PERSON_PENALTY = 1e8
EDGE_PENALTY = 1e6


# ---------------------------------------------------------------------------
# Cost + EMA (GPU and CPU)
# ---------------------------------------------------------------------------

def compute_cost_and_ema_gpu(warped_a_t, warped_b_t, overlap_in_bbox_t,
                             cost_ema_t, ema_alpha, person_mask_bbox_t,
                             fg_mask_bbox_t, fg_penalty, person_penalty,
                             overlap_bbox):
    """
    GPU cost + EMA + penalty injection.

    Penalty hierarchy (additive on cost_ema):
        photometric (0 - 1e5)
        + fg_penalty (default 5e7) where fg_mask AND NOT person_mask
        + person_penalty (default 1e8) where person_mask

    Returns (updated cost_ema_t, cost_for_dp as a numpy float array).
    """
    x0, y0, x1, y1 = overlap_bbox
    wa_bb = warped_a_t[0, :, y0:y1, x0:x1].float()
    wb_bb = warped_b_t[0, :, y0:y1, x0:x1].float()

    diff = wa_bb - wb_bb
    photo_cost = (diff * diff).sum(dim=0)

    overlap_mask = overlap_in_bbox_t > 0
    photo_cost = torch.where(overlap_mask, photo_cost,
                             torch.tensor(1e9, device=photo_cost.device,
                                          dtype=photo_cost.dtype))

    if cost_ema_t is None or cost_ema_t.shape != photo_cost.shape:
        cost_ema_t = photo_cost.clone()
    else:
        if ema_alpha >= 1.0:
            cost_ema_t = photo_cost.clone()
        else:
            cost_ema_t = ema_alpha * photo_cost + (1.0 - ema_alpha) * cost_ema_t

    cost_for_dp = cost_ema_t.clone()

    # FG penalty first (lower priority), person second (higher).
    # A pixel that's both gets the sum, but person (1e8) >> fg (5e7) so
    # the effect is the same as taking the max.
    if fg_mask_bbox_t is not None:
        if person_mask_bbox_t is not None:
            fg_only = fg_mask_bbox_t & (~person_mask_bbox_t)
        else:
            fg_only = fg_mask_bbox_t
        cost_for_dp = torch.where(fg_only > 0,
                                  cost_for_dp + fg_penalty,
                                  cost_for_dp)
    if person_mask_bbox_t is not None:
        cost_for_dp = torch.where(person_mask_bbox_t > 0,
                                  cost_for_dp + person_penalty,
                                  cost_for_dp)

    cost_for_dp_cpu = cost_for_dp.cpu().numpy()
    return cost_ema_t, cost_for_dp_cpu


def compute_cost_fast_cpu(wa_bb, wb_bb, overlap_in_bbox, cost_scratch):
    """
    CPU photometric cost: sum of squared BGR differences over the
    overlap bbox. Out-of-overlap pixels are marked with a 1e9 sentinel
    so the DP will never route through them.
    """
    diff = cv2.absdiff(wa_bb, wb_bb)
    diff_f = diff.astype(np.float32, copy=False)
    np.multiply(diff_f, diff_f, out=cost_scratch)
    cost = cost_scratch.sum(axis=2)
    cost[overlap_in_bbox == 0] = 1e9
    return cost


# ---------------------------------------------------------------------------
# DP seam + utilities
# ---------------------------------------------------------------------------

def find_dp_seam(cost):
    """
    Find the minimum-cost top-to-bottom path through `cost` (shape H, W).
    Each row's seam pixel is within 1 of the row below's seam pixel
    (the standard 3-neighbor DP).
    """
    H, W = cost.shape
    dp = cost.copy()
    for y in range(1, H):
        prev = dp[y - 1]
        left  = np.concatenate(([np.inf], prev[:-1]))
        right = np.concatenate((prev[1:], [np.inf]))
        dp[y] += np.minimum(np.minimum(prev, left), right)
    seam_x = np.empty(H, dtype=np.int32)
    seam_x[-1] = int(np.argmin(dp[-1]))
    for y in range(H - 2, -1, -1):
        x = seam_x[y + 1]
        x0 = max(x - 1, 0)
        x1 = min(x + 2, W)
        local = dp[y, x0:x1]
        seam_x[y] = x0 + int(np.argmin(local))
    return seam_x


def upscale_seam(seam_x_small, bbox_shape, downscale):
    """Interpolate a downscaled seam (height H_small) back to the full bbox
    height, then scale x coords by `downscale`."""
    H_bb, W_bb = bbox_shape
    H_small = seam_x_small.shape[0]
    ys_small = np.arange(H_small, dtype=np.float32)
    ys_full  = np.linspace(0, H_small - 1, H_bb, dtype=np.float32)
    seam_x_full = np.interp(ys_full, ys_small, seam_x_small.astype(np.float32))
    seam_x_full = (seam_x_full * downscale).astype(np.int32)
    return np.clip(seam_x_full, 0, W_bb - 1)


def add_edge_margin_penalty(cost, margin, edge_penalty=EDGE_PENALTY):
    """In-place: add `edge_penalty` to the leftmost / rightmost `margin`
    columns of the cost map. Keeps the seam from grazing the bbox edges
    where the multi-band blur would reach into padded pixels."""
    if margin <= 0:
        return
    margin = min(margin, cost.shape[1] // 2)
    cost[:,  :margin]  += edge_penalty
    cost[:, -margin:] += edge_penalty


def add_seam_regularizer(cost, seam_prev_small, lam):
    """In-place: add a quadratic attractor toward the previous frame's seam
    (`lam * (x - seam_prev(y))^2`). Stabilizes the seam from frame to
    frame; too high and the seam reacts sluggishly to moving people."""
    if seam_prev_small is None or lam <= 0:
        return
    H, W = cost.shape
    col_idx = np.arange(W, dtype=np.float32)[None, :]
    seam_prev_col = seam_prev_small.astype(np.float32)[:, None]
    dx = col_idx - seam_prev_col
    penalty = (dx * dx) * float(lam)
    cost += penalty
