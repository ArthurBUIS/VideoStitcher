"""
Multi-band Laplacian-pyramid blending around the DP seam.

`build_soft_mask_fast` produces the soft alpha mask that drives the
blend; the GPU and CPU compositors use the same alpha but different
pyramid implementations (PyTorch conv2d on GPU, cv2.pyrDown/Up on CPU).
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Soft mask from the DP seam (shared by GPU and CPU composite)
# ---------------------------------------------------------------------------

def build_soft_mask_fast(seam_x_full, bbox_shape, static, blend_width):
    """
    Build a soft alpha mask (float32 in [0,1]) from the DP seam.

    Hard mask: 1 to the left of the seam, 0 to the right. Then smoothed
    horizontally to a Gaussian of effective sigma = blend_width/3 via a
    coarse-resolution Gaussian blur on a downsampled pyramid level
    (cheaper than blurring at full resolution for large blend widths).
    Finally, force the "only A" region to 1 and "only B" to 0 so the
    blend doesn't leak across hard edges.
    """
    H_bb, W_bb = bbox_shape
    col_idx = np.arange(W_bb, dtype=np.int32)[None, :]
    seam_col = seam_x_full[:, None]
    hard = (col_idx < seam_col).astype(np.float32)

    target_sigma = max(1.0, blend_width / 3.0)
    depth = int(np.floor(np.log2(target_sigma / 3.0)))
    depth = max(0, min(depth, 4))

    cur = hard
    for _ in range(depth):
        cur = cv2.pyrDown(cur)
    coarse_sigma = target_sigma / (2 ** depth)
    ks = max(3, int(6 * coarse_sigma) | 1)
    cur = cv2.GaussianBlur(cur, (ks, ks), sigmaX=coarse_sigma,
                           sigmaY=coarse_sigma)

    shapes = [(H_bb, W_bb)]
    for _ in range(depth):
        ph, pw = shapes[-1]
        shapes.append(((ph + 1) // 2, (pw + 1) // 2))
    for i in range(depth, 0, -1):
        th, tw = shapes[i - 1]
        cur = cv2.pyrUp(cur, dstsize=(tw, th))

    soft = cur
    only_a = static["only_a_in_bbox"]
    only_b = static["only_b_in_bbox"]
    soft[only_a > 0] = 1.0
    soft[only_b > 0] = 0.0
    return soft


# ---------------------------------------------------------------------------
# GPU pyramid + composite
# ---------------------------------------------------------------------------

_PYR_KERNEL_1D = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32) / 16.0


def _get_pyr_kernel_2d(device):
    k1 = _PYR_KERNEL_1D.to(device)
    k2 = k1[:, None] * k1[None, :]
    return k2[None, None, :, :]


def _pyr_down_torch(x, kernel2d):
    """
    x: (N, C, H, W). Returns (N, C, H//2, W//2).

    Per-channel Gaussian blur (5x5 binomial kernel, grouped conv) +
    stride-2 downsample. Batch + channel layout means a single conv2d
    handles every image and channel together.
    """
    _, C, _, _ = x.shape
    x_pad = F.pad(x, (2, 2, 2, 2), mode="replicate")
    kern = kernel2d.expand(C, 1, 5, 5)
    blurred = F.conv2d(x_pad, kern, groups=C)
    return blurred[:, :, ::2, ::2]


def _pyr_up_torch(x, target_hw, kernel2d):
    """
    x: (N, C, Hs, Ws). Returns (N, C, target_h, target_w).

    Zero-insert at every other location, Gaussian smooth (scaled by 4
    to compensate for the zeros), then crop to the requested target
    shape. Batched in (N, C, H, W) so one conv2d call handles every
    image and channel.
    """
    N, C, Hs, Ws = x.shape
    up = x.new_zeros((N, C, Hs * 2, Ws * 2))
    up[:, :, ::2, ::2] = x
    up_pad = F.pad(up, (2, 2, 2, 2), mode="replicate")
    kern = (kernel2d * 4.0).expand(C, 1, 5, 5)
    blurred = F.conv2d(up_pad, kern, groups=C)
    Th, Tw = target_hw
    return blurred[:, :, :Th, :Tw]


def _build_gaussian_pyramid_torch(x, levels, kernel2d):
    """x: (N, C, H, W) → list[(N, C, H_i, W_i)] of length levels+1."""
    gp = [x]
    for _ in range(levels):
        gp.append(_pyr_down_torch(gp[-1], kernel2d))
    return gp


def _build_laplacian_pyramid_torch(x, levels, kernel2d):
    """x: (N, C, H, W) → list of laplacian levels (length levels+1)."""
    gp = _build_gaussian_pyramid_torch(x, levels, kernel2d)
    lp = []
    for i in range(levels):
        up = _pyr_up_torch(gp[i + 1], gp[i].shape[2:], kernel2d)
        lp.append(gp[i] - up)
    lp.append(gp[levels])
    return lp


def _reconstruct_from_laplacian_torch(lp, kernel2d):
    img = lp[-1]
    for level in reversed(lp[:-1]):
        img = _pyr_up_torch(img, level.shape[2:], kernel2d)
        img = img + level
    return img


def _composite_to_gpu_tensor(warped_a_t, warped_b_t, static, seam_x_full,
                              blend_width, blend_levels, gpu_ctx):
    """
    Shared multi-band Laplacian blend on GPU.

    The multi-band blend only affects pixels within ±blend_width of the
    seam — outside that strip the soft mask is hard 0 or 1 and the
    blend reduces to a copy. Building the pyramid on the entire overlap
    bbox is wasteful; instead we:

      1. Compute strip = [seam.min() - blend_width, seam.max() + blend_width]
         clipped to the bbox.
      2. Hard-copy A on the overlap region left of the strip, B on the
         overlap region right. only_A / only_B regions are filled
         output-wide first.
      3. Multi-band only on the strip (typically 4-5x smaller than the
         full bbox in x).

    Returns out_t as a (3, H_out, W_out) uint8 tensor on GPU. When
    --autocrop is on, (H_out, W_out) is the autocrop dimensions; when
    off it's the full canvas. The caller is responsible for any host
    transfer + synchronisation.
    """
    device = gpu_ctx["device"]
    kernel2d = gpu_ctx["kernel2d"]
    x0, y0, x1, y1 = static["overlap_bbox"]
    H_bb = y1 - y0
    W_bb = x1 - x0

    # Strip x-range in bbox-local coords.
    seam_min = int(seam_x_full.min())
    seam_max = int(seam_x_full.max())
    x_strip_min = max(0, seam_min - blend_width)
    x_strip_max = min(W_bb, seam_max + blend_width + 1)
    strip_w = x_strip_max - x_strip_min

    # Stay channel-first (C, H, W) for the whole composite to avoid two
    # full-output .permute(1, 2, 0).contiguous() copies of the warped
    # frames. Pyramid math expects channel-first anyway; the layout swap
    # only matters for the final cv2 output, so we permute once at the
    # very end onto the (H, W, 3) pinned host buffer.
    a_t = warped_a_t[0]   # (3, H_out, W_out) — view, no copy
    b_t = warped_b_t[0]

    H_out = a_t.shape[1]
    W_out = a_t.shape[2]
    out_t = torch.zeros((3, H_out, W_out), dtype=torch.uint8, device=device)

    # Step 1: hard-copy only_A and only_B output-wide. Masks are (H, W);
    # unsqueeze(0) broadcasts across the 3-channel dim of the tensors.
    only_a_m = (gpu_ctx["only_a_u8_t"] > 0).unsqueeze(0)
    only_b_m = (gpu_ctx["only_b_u8_t"] > 0).unsqueeze(0)
    out_t = torch.where(only_a_m, a_t, out_t)
    out_t = torch.where(only_b_m, b_t, out_t)

    # Step 2: in the overlap, hard-copy A left of the strip and B right.
    overlap_bb_t = gpu_ctx["overlap_in_bbox_t"]
    if x_strip_min > 0:
        left_overlap = (overlap_bb_t[:, :x_strip_min] > 0).unsqueeze(0)
        a_left = a_t[:, y0:y1, x0:x0 + x_strip_min]
        cur = out_t[:, y0:y1, x0:x0 + x_strip_min]
        out_t[:, y0:y1, x0:x0 + x_strip_min] = torch.where(left_overlap, a_left, cur)
    if x_strip_max < W_bb:
        right_overlap = (overlap_bb_t[:, x_strip_max:] > 0).unsqueeze(0)
        b_right = b_t[:, y0:y1, x0 + x_strip_max:x1]
        cur = out_t[:, y0:y1, x0 + x_strip_max:x1]
        out_t[:, y0:y1, x0 + x_strip_max:x1] = torch.where(right_overlap, b_right, cur)

    # Step 3: multi-band on the strip. Strip slices are also channel-first.
    a_strip_t = a_t[:, y0:y1, x0 + x_strip_min:x0 + x_strip_max].contiguous()
    b_strip_t = b_t[:, y0:y1, x0 + x_strip_min:x0 + x_strip_max].contiguous()

    only_a_strip = (gpu_ctx["only_a_in_bbox_t"][:, x_strip_min:x_strip_max]
                    > 0).unsqueeze(0)
    only_b_strip = (gpu_ctx["only_b_in_bbox_t"][:, x_strip_min:x_strip_max]
                    > 0).unsqueeze(0)
    a_strip_filled = torch.where(only_b_strip, b_strip_t, a_strip_t)
    b_strip_filled = torch.where(only_a_strip, a_strip_t, b_strip_t)

    strip_static = {
        "only_a_in_bbox": static["only_a_in_bbox"][:, x_strip_min:x_strip_max],
        "only_b_in_bbox": static["only_b_in_bbox"][:, x_strip_min:x_strip_max],
    }
    seam_x_strip = (seam_x_full.astype(np.int32) - x_strip_min)
    seam_x_strip = np.clip(seam_x_strip, 0, strip_w - 1)
    mask_strip_np = build_soft_mask_fast(
        seam_x_strip, (H_bb, strip_w), strip_static, blend_width,
    )
    mask_strip_t = torch.from_numpy(mask_strip_np).to(device, non_blocking=True)

    # Batch A and B together so each pyramid level is built with ONE
    # conv2d kernel launch per pyrDown/pyrUp instead of two. ab_f has
    # shape (2, 3, H_bb, strip_w); kernel halve per-level fused across
    # both images. Mask stays (1, 1, H, W) since it has different
    # channel count.
    a_f = a_strip_filled.float().unsqueeze(0)            # (1, 3, H, W)
    b_f = b_strip_filled.float().unsqueeze(0)            # (1, 3, H, W)
    ab_f = torch.cat([a_f, b_f], dim=0)                  # (2, 3, H, W)
    m_f = mask_strip_t.unsqueeze(0).unsqueeze(0)         # (1, 1, H, W)

    min_dim = min(H_bb, strip_w)
    max_levels = max(1, int(np.log2(min_dim)) - 2)
    levels = min(blend_levels, max_levels)

    lp_ab = _build_laplacian_pyramid_torch(ab_f, levels, kernel2d)
    gp_m = _build_gaussian_pyramid_torch(m_f, levels, kernel2d)

    # torch.lerp(start, end, weight) = start + weight * (end - start)
    # so lerp(B, A, gm) = B + gm*(A - B) = A*gm + B*(1-gm). One fused
    # op per level instead of two muls + one add.
    blended_lp = [torch.lerp(lp_ab[i][1:2], lp_ab[i][0:1], gp_m[i])
                  for i in range(len(lp_ab))]
    recon = _reconstruct_from_laplacian_torch(blended_lp, kernel2d).clamp(0, 255)
    blended_strip_t = recon[0].to(torch.uint8).contiguous()  # (3, H_bb, strip_w)

    valid_strip = (gpu_ctx["valid_in_bbox_t"][:, x_strip_min:x_strip_max]
                   > 0).unsqueeze(0)
    cur = out_t[:, y0:y1, x0 + x_strip_min:x0 + x_strip_max]
    out_t[:, y0:y1, x0 + x_strip_min:x0 + x_strip_max] = torch.where(
        valid_strip, blended_strip_t, cur,
    )

    return out_t


def composite_multiband_gpu_async(warped_a_t, warped_b_t, static, seam_x_full,
                                   blend_width, blend_levels,
                                   pinned_buf, gpu_ctx):
    """
    Asynchronous GPU composite.

    Like composite_multiband_gpu_resident, but:
      - Writes the final (H, W, 3) uint8 frame into the provided
        pinned_buf instead of an out_buf scratch numpy array. The
        warped frames + masks are already sized to the rendering target
        (the autocrop rect when --autocrop is on, otherwise the full
        canvas) — the crop translation is baked into the homographies
        upstream of the warp, so there is no separate crop step here.
      - Does NOT synchronize before returning. Returns a torch.cuda.Event
        recorded after the pinned copy.

    The downstream consumer (writer thread) is responsible for waiting
    on the event before reading pinned_buf. This frees the composite
    worker to immediately launch the next frame's kernels, keeping the
    GPU's stream queue fuller and overlapping with the writer's encode.
    """
    out_t = _composite_to_gpu_tensor(
        warped_a_t, warped_b_t, static, seam_x_full,
        blend_width, blend_levels, gpu_ctx,
    )
    out_hwc = out_t.permute(1, 2, 0).contiguous()
    pinned_buf.copy_(out_hwc, non_blocking=True)
    event = torch.cuda.Event()
    event.record()
    return event


# ---------------------------------------------------------------------------
# CPU pyramid + composite
# ---------------------------------------------------------------------------

def fill_invalid_with_other_cpu(a_bb_u8, b_bb_u8, static):
    """Pre-fill A's 'only B' region with B (and vice versa) before the
    Laplacian pyramid build, so pyrDown doesn't pull in zeros at hard
    edges."""
    only_a = static["only_a_in_bbox"]
    only_b = static["only_b_in_bbox"]
    a_out = a_bb_u8.copy()
    b_out = b_bb_u8.copy()
    cv2.copyTo(b_bb_u8, only_b, a_out)
    cv2.copyTo(a_bb_u8, only_a, b_out)
    return a_out, b_out


def build_gaussian_pyramid_cpu(img_f32, levels):
    gp = [img_f32]
    for _ in range(levels):
        gp.append(cv2.pyrDown(gp[-1]))
    return gp


def build_laplacian_pyramid_cpu(img_f32, levels):
    gp = build_gaussian_pyramid_cpu(img_f32, levels)
    lp = []
    for i in range(levels):
        up = cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
        lp.append(gp[i] - up)
    lp.append(gp[levels])
    return lp


def reconstruct_from_laplacian_cpu(lp):
    img = lp[-1]
    for level in reversed(lp[:-1]):
        img = cv2.pyrUp(img, dstsize=(level.shape[1], level.shape[0]))
        img = img + level
    return img


def blend_pyramids_fast_cpu(lp_a, lp_b, gp_m):
    blended = []
    for la, lb, gm in zip(lp_a, lp_b, gp_m):
        if la.ndim == 3:
            if gm.ndim == 2:
                gm3 = cv2.merge([gm] * la.shape[2])
            elif gm.ndim == 3 and gm.shape[2] == 1:
                gm3 = cv2.merge([gm[:, :, 0]] * la.shape[2])
            else:
                gm3 = gm
        else:
            gm3 = gm
        la_gm = cv2.multiply(la, gm3)
        one_minus = np.empty_like(gm3)
        np.subtract(1.0, gm3, out=one_minus)
        lb_gm = cv2.multiply(lb, one_minus)
        out = cv2.add(la_gm, lb_gm)
        blended.append(out)
    return blended


def composite_multiband_cpu(warped_a, warped_b, static, seam_x_full,
                            blend_width, blend_levels, out_buf):
    """CPU multi-band Laplacian blend using cv2.pyrDown / pyrUp."""
    x0, y0, x1, y1 = static["overlap_bbox"]
    out_buf.fill(0)
    cv2.copyTo(warped_a, static["only_a_u8"], out_buf)
    cv2.copyTo(warped_b, static["only_b_u8"], out_buf)
    H_bb = y1 - y0
    W_bb = x1 - x0
    bbox_shape = (H_bb, W_bb)
    a_bb = warped_a[y0:y1, x0:x1]
    b_bb = warped_b[y0:y1, x0:x1]
    a_bb_pad, b_bb_pad = fill_invalid_with_other_cpu(a_bb, b_bb, static)
    mask_f32 = build_soft_mask_fast(seam_x_full, bbox_shape, static, blend_width)
    min_dim = min(a_bb_pad.shape[:2])
    max_levels = max(1, int(np.log2(min_dim)) - 2)
    levels = min(blend_levels, max_levels)
    a_f = a_bb_pad.astype(np.float32)
    b_f = b_bb_pad.astype(np.float32)
    lp_a = build_laplacian_pyramid_cpu(a_f, levels)
    lp_b = build_laplacian_pyramid_cpu(b_f, levels)
    gp_m = build_gaussian_pyramid_cpu(mask_f32, levels)
    blended_lp = blend_pyramids_fast_cpu(lp_a, lp_b, gp_m)
    recon = reconstruct_from_laplacian_cpu(blended_lp)
    np.clip(recon, 0, 255, out=recon)
    blended_bb = recon.astype(np.uint8)
    valid_in_bbox = cv2.bitwise_or(static["mask_a_in_bbox"],
                                   static["mask_b_in_bbox"])
    cv2.copyTo(blended_bb, valid_in_bbox, out_buf[y0:y1, x0:x1])
    return out_buf


def get_pyr_kernel_2d(device):
    """Public wrapper around _get_pyr_kernel_2d for use from the pipeline."""
    return _get_pyr_kernel_2d(device)


# ---------------------------------------------------------------------------
# 3-camera composite
# ---------------------------------------------------------------------------
#
# The 3-camera composite runs two pairwise multi-band blends (L<>C and
# C<>R) into a single output tensor. With the x_mid seam cap (see
# stitcher.seam.add_x_mid_seam_cap) plus a blend_width margin, the two
# blend strips are guaranteed spatially disjoint on the canvas, so the
# second pair-blend can never overwrite pixels the first one wrote.
#
# Layout of the final canvas (left-to-right) after the composite:
#
#   [ only-L ][ L<>C blend strip ][ Center, possibly triple-overlap ]
#                                 [ C<>R blend strip ][ only-R ]
#
# In a strict only-C band between the two blend strips (if it exists),
# both pairwise blends naturally produce ~Center because their soft
# alphas have already tapered out -- so even when the L<>C and C<>R
# OVERLAP bboxes touch (which they typically do on real portal rigs),
# the result is consistent.


def _blend_pair_strip_gpu(out_t, a_t, b_t,
                          overlap_dict, overlap_gpu, seam_x_full,
                          blend_width, blend_levels, device, kernel2d):
    """
    One pairwise multi-band blend, written into an EXISTING out_t.

    Mirrors steps 2 and 3 of _composite_to_gpu_tensor (the hard-copy
    of a/b on either side of the strip, plus the multi-band blend
    inside the strip) for one pair of warped tensors. Does NOT touch
    the canvas-wide only_a / only_b regions -- the caller handles
    those once for all three cameras.

    a_t, b_t        : (3, H_out, W_out) views of the two warped tensors
                       (typically warped_L_t[0] and warped_C_t[0] for
                       the L<>C call; warped_C_t[0] and warped_R_t[0]
                       for the C<>R call).
    overlap_dict    : output of OverlapContext.as_static_dict() -- has
                       overlap_bbox + only_a_in_bbox + only_b_in_bbox
                       in the same keys the 2-camera function reads.
    overlap_gpu     : dict with GPU tensors for this overlap:
                       'overlap_in_bbox_t', 'only_a_in_bbox_t',
                       'only_b_in_bbox_t', 'valid_in_bbox_t'. Built
                       once at startup in the pipeline.
    seam_x_full     : (H_bb,) int seam path on this overlap's bbox.
    """
    x0, y0, x1, y1 = overlap_dict["overlap_bbox"]
    H_bb = y1 - y0
    W_bb = x1 - x0

    seam_min = int(seam_x_full.min())
    seam_max = int(seam_x_full.max())
    x_strip_min = max(0, seam_min - blend_width)
    x_strip_max = min(W_bb, seam_max + blend_width + 1)
    strip_w = x_strip_max - x_strip_min

    overlap_bb_t = overlap_gpu["overlap_in_bbox_t"]
    if x_strip_min > 0:
        left_overlap = (overlap_bb_t[:, :x_strip_min] > 0).unsqueeze(0)
        a_left = a_t[:, y0:y1, x0:x0 + x_strip_min]
        cur = out_t[:, y0:y1, x0:x0 + x_strip_min]
        out_t[:, y0:y1, x0:x0 + x_strip_min] = torch.where(
            left_overlap, a_left, cur,
        )
    if x_strip_max < W_bb:
        right_overlap = (overlap_bb_t[:, x_strip_max:] > 0).unsqueeze(0)
        b_right = b_t[:, y0:y1, x0 + x_strip_max:x1]
        cur = out_t[:, y0:y1, x0 + x_strip_max:x1]
        out_t[:, y0:y1, x0 + x_strip_max:x1] = torch.where(
            right_overlap, b_right, cur,
        )

    a_strip_t = a_t[:, y0:y1, x0 + x_strip_min:x0 + x_strip_max].contiguous()
    b_strip_t = b_t[:, y0:y1, x0 + x_strip_min:x0 + x_strip_max].contiguous()

    only_a_strip = (overlap_gpu["only_a_in_bbox_t"][:, x_strip_min:x_strip_max]
                    > 0).unsqueeze(0)
    only_b_strip = (overlap_gpu["only_b_in_bbox_t"][:, x_strip_min:x_strip_max]
                    > 0).unsqueeze(0)
    a_strip_filled = torch.where(only_b_strip, b_strip_t, a_strip_t)
    b_strip_filled = torch.where(only_a_strip, a_strip_t, b_strip_t)

    strip_static = {
        "only_a_in_bbox": overlap_dict["only_a_in_bbox"][:, x_strip_min:x_strip_max],
        "only_b_in_bbox": overlap_dict["only_b_in_bbox"][:, x_strip_min:x_strip_max],
    }
    seam_x_strip = (seam_x_full.astype(np.int32) - x_strip_min)
    seam_x_strip = np.clip(seam_x_strip, 0, strip_w - 1)
    mask_strip_np = build_soft_mask_fast(
        seam_x_strip, (H_bb, strip_w), strip_static, blend_width,
    )
    mask_strip_t = torch.from_numpy(mask_strip_np).to(device, non_blocking=True)

    a_f = a_strip_filled.float().unsqueeze(0)
    b_f = b_strip_filled.float().unsqueeze(0)
    ab_f = torch.cat([a_f, b_f], dim=0)
    m_f = mask_strip_t.unsqueeze(0).unsqueeze(0)

    min_dim = min(H_bb, strip_w)
    max_levels = max(1, int(np.log2(min_dim)) - 2)
    levels = min(blend_levels, max_levels)

    lp_ab = _build_laplacian_pyramid_torch(ab_f, levels, kernel2d)
    gp_m = _build_gaussian_pyramid_torch(m_f, levels, kernel2d)

    blended_lp = [torch.lerp(lp_ab[i][1:2], lp_ab[i][0:1], gp_m[i])
                  for i in range(len(lp_ab))]
    recon = _reconstruct_from_laplacian_torch(blended_lp, kernel2d).clamp(0, 255)
    blended_strip_t = recon[0].to(torch.uint8).contiguous()

    valid_strip = (overlap_gpu["valid_in_bbox_t"][:, x_strip_min:x_strip_max]
                   > 0).unsqueeze(0)
    cur = out_t[:, y0:y1, x0 + x_strip_min:x0 + x_strip_max]
    out_t[:, y0:y1, x0 + x_strip_min:x0 + x_strip_max] = torch.where(
        valid_strip, blended_strip_t, cur,
    )
    return out_t


def _composite_to_gpu_tensor_3cam(
    warped_L_t, warped_C_t, warped_R_t,
    static_3cam,
    seam_LC_x_full, seam_CR_x_full,
    blend_width, blend_levels, gpu_ctx_3cam,
):
    """
    GPU composite for the 3-camera path. Same data flow as
    _composite_to_gpu_tensor, but applied as two pairwise blends.

    static_3cam is the dict returned by
    geometry.build_static_geometry_3cam (carries overlap_LC,
    overlap_CR, only_L_u8, only_C_u8, only_R_u8, x_mid).

    gpu_ctx_3cam is the 3-camera analogue of gpu_ctx -- carries the
    device + kernel + the canvas-wide only_*_u8_t tensors PLUS a
    nested per-overlap dict for the L<>C and C<>R contexts:

        gpu_ctx_3cam = {
            'device': ..., 'kernel2d': ...,
            'only_L_u8_t': ..., 'only_C_u8_t': ..., 'only_R_u8_t': ...,
            'overlap_LC_gpu': {
                'overlap_in_bbox_t', 'only_a_in_bbox_t',
                'only_b_in_bbox_t', 'valid_in_bbox_t',
            },
            'overlap_CR_gpu': { ... same shape ... },
        }
    """
    device = gpu_ctx_3cam["device"]
    kernel2d = gpu_ctx_3cam["kernel2d"]

    L_t = warped_L_t[0]
    C_t = warped_C_t[0]
    R_t = warped_R_t[0]

    H_out = L_t.shape[1]
    W_out = L_t.shape[2]
    out_t = torch.zeros((3, H_out, W_out), dtype=torch.uint8, device=device)

    # Step 1: hard-copy each camera's "only this camera sees" region.
    only_L_m = (gpu_ctx_3cam["only_L_u8_t"] > 0).unsqueeze(0)
    only_C_m = (gpu_ctx_3cam["only_C_u8_t"] > 0).unsqueeze(0)
    only_R_m = (gpu_ctx_3cam["only_R_u8_t"] > 0).unsqueeze(0)
    out_t = torch.where(only_L_m, L_t, out_t)
    out_t = torch.where(only_C_m, C_t, out_t)
    out_t = torch.where(only_R_m, R_t, out_t)

    # Step 2: L<>C pairwise blend on the L<>C bbox + seam_LC.
    out_t = _blend_pair_strip_gpu(
        out_t, L_t, C_t,
        static_3cam["overlap_LC"].as_static_dict(),
        gpu_ctx_3cam["overlap_LC_gpu"],
        seam_LC_x_full,
        blend_width, blend_levels, device, kernel2d,
    )

    # Step 3: C<>R pairwise blend on the C<>R bbox + seam_CR.
    # Spatially disjoint from step 2 by the x_mid + blend_width cap,
    # so no overwrite of step 2's blend pixels.
    out_t = _blend_pair_strip_gpu(
        out_t, C_t, R_t,
        static_3cam["overlap_CR"].as_static_dict(),
        gpu_ctx_3cam["overlap_CR_gpu"],
        seam_CR_x_full,
        blend_width, blend_levels, device, kernel2d,
    )

    return out_t


def composite_multiband_gpu_async_3cam(
    warped_L_t, warped_C_t, warped_R_t,
    static_3cam,
    seam_LC_x_full, seam_CR_x_full,
    blend_width, blend_levels,
    pinned_buf, gpu_ctx_3cam,
):
    """
    Asynchronous 3-camera GPU composite. Mirrors
    composite_multiband_gpu_async: builds the output tensor, copies
    into the supplied pinned host buffer, returns a CUDA event the
    writer thread synchronises on.
    """
    out_t = _composite_to_gpu_tensor_3cam(
        warped_L_t, warped_C_t, warped_R_t, static_3cam,
        seam_LC_x_full, seam_CR_x_full, blend_width, blend_levels,
        gpu_ctx_3cam,
    )
    out_hwc = out_t.permute(1, 2, 0).contiguous()
    pinned_buf.copy_(out_hwc, non_blocking=True)
    event = torch.cuda.Event()
    event.record()
    return event


def _blend_pair_strip_cpu(out_buf, warped_a, warped_b,
                          overlap_dict, seam_x_full,
                          blend_width, blend_levels):
    """
    CPU analogue of _blend_pair_strip_gpu. Writes one pairwise
    multi-band blend into out_buf. Does NOT touch only_X canvas
    regions (the 3-cam caller handles those once for all three).
    """
    x0, y0, x1, y1 = overlap_dict["overlap_bbox"]
    H_bb = y1 - y0
    W_bb = x1 - x0
    bbox_shape = (H_bb, W_bb)
    a_bb = warped_a[y0:y1, x0:x1]
    b_bb = warped_b[y0:y1, x0:x1]
    a_bb_pad, b_bb_pad = fill_invalid_with_other_cpu(a_bb, b_bb, overlap_dict)

    mask_f32 = build_soft_mask_fast(
        seam_x_full, bbox_shape, overlap_dict, blend_width,
    )
    min_dim = min(a_bb_pad.shape[:2])
    max_levels = max(1, int(np.log2(min_dim)) - 2)
    levels = min(blend_levels, max_levels)

    a_f = a_bb_pad.astype(np.float32)
    b_f = b_bb_pad.astype(np.float32)
    lp_a = build_laplacian_pyramid_cpu(a_f, levels)
    lp_b = build_laplacian_pyramid_cpu(b_f, levels)
    gp_m = build_gaussian_pyramid_cpu(mask_f32, levels)
    blended_lp = blend_pyramids_fast_cpu(lp_a, lp_b, gp_m)
    recon = reconstruct_from_laplacian_cpu(blended_lp)
    np.clip(recon, 0, 255, out=recon)
    blended_bb = recon.astype(np.uint8)
    valid_in_bbox = cv2.bitwise_or(
        overlap_dict["mask_a_in_bbox"],
        overlap_dict["mask_b_in_bbox"],
    )
    cv2.copyTo(blended_bb, valid_in_bbox, out_buf[y0:y1, x0:x1])
    return out_buf


def composite_multiband_cpu_3cam(
    warped_L, warped_C, warped_R,
    static_3cam,
    seam_LC_x_full, seam_CR_x_full,
    blend_width, blend_levels, out_buf,
):
    """
    CPU 3-camera composite. Sequenced like the GPU version:
      1. zero out_buf
      2. hard-copy only_L, only_C, only_R (cv2.copyTo with canvas masks)
      3. L<>C pairwise multi-band blend on its bbox
      4. C<>R pairwise multi-band blend on its bbox
    """
    out_buf.fill(0)
    cv2.copyTo(warped_L, static_3cam["only_L_u8"], out_buf)
    cv2.copyTo(warped_C, static_3cam["only_C_u8"], out_buf)
    cv2.copyTo(warped_R, static_3cam["only_R_u8"], out_buf)

    out_buf = _blend_pair_strip_cpu(
        out_buf, warped_L, warped_C,
        static_3cam["overlap_LC"].as_static_dict(),
        seam_LC_x_full, blend_width, blend_levels,
    )
    out_buf = _blend_pair_strip_cpu(
        out_buf, warped_C, warped_R,
        static_3cam["overlap_CR"].as_static_dict(),
        seam_CR_x_full, blend_width, blend_levels,
    )
    return out_buf
