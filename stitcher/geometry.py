"""
Homography, canvas, remap tables, static per-camera geometry, autocrop.

Everything here runs once at startup. The outputs are reused unchanged
for every frame.
"""

import cv2
import numpy as np


def estimate_homography(img_a, img_b):
    """ORB + RANSAC. Returns H mapping B's coordinates into A's frame."""
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=4000)
    kpts_a, desc_a = orb.detectAndCompute(gray_a, None)
    kpts_b, desc_b = orb.detectAndCompute(gray_b, None)
    if desc_a is None or desc_b is None:
        raise RuntimeError("Could not detect ORB features.")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc_b, desc_a)
    matches = sorted(matches, key=lambda m: m.distance)
    matches = matches[: max(50, len(matches) // 2)]
    pts_b = np.float32([kpts_b[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_a = np.float32([kpts_a[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("Homography estimation failed.")
    return H


def compute_canvas(shape_a, shape_b, H_b_to_a):
    """
    Compute the panorama canvas size and the translation that brings A's
    top-left to canvas (0, 0).

    Returns (canvas_size, T, H_b_to_canvas, H_a_to_canvas) where
    canvas_size = (width, height) and T is the 3x3 translation matrix.
    """
    h_a, w_a = shape_a[:2]
    h_b, w_b = shape_b[:2]
    corners_b = np.float32([[0, 0], [w_b, 0], [w_b, h_b], [0, h_b]]).reshape(-1, 1, 2)
    corners_b_in_a = cv2.perspectiveTransform(corners_b, H_b_to_a)
    corners_a = np.float32([[0, 0], [w_a, 0], [w_a, h_a], [0, h_a]]).reshape(-1, 1, 2)
    all_corners = np.concatenate([corners_a, corners_b_in_a], axis=0)
    x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)
    tx, ty = -x_min, -y_min
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    H_b_to_canvas = T @ H_b_to_a
    H_a_to_canvas = T.copy()
    return (canvas_w, canvas_h), T, H_b_to_canvas, H_a_to_canvas


def find_autocrop_rect(H_b_to_a, shape_a, shape_b, canvas_size, T):
    """
    Build a rectangle from B's two right corners (warped to canvas) and A's
    left edge. The right side of the rectangle is the more-conservative
    (smaller-x) of B's two right corners; the rectangle spans the y-range
    formed by both B right corners; the left side is A's left edge on canvas.

    Returns (x, y, w, h) in canvas coords.
    """
    canvas_w, canvas_h = canvas_size
    h_a, w_a = shape_a[:2]
    h_b, w_b = shape_b[:2]
    tx, ty = float(T[0, 2]), float(T[1, 2])

    # B's right corners after warp to canvas (top_right = warped (w_b, 0),
    # bottom_right = warped (w_b, h_b)).
    b_right_src = np.float32([[w_b, 0], [w_b, h_b]]).reshape(-1, 1, 2)
    b_right_canvas = cv2.perspectiveTransform(b_right_src, H_b_to_a).reshape(-1, 2)
    b_right_canvas += np.array([tx, ty], dtype=np.float32)
    top_right, bottom_right = b_right_canvas[0], b_right_canvas[1]

    # Pick the one with the lower x as (x_1, y_1); the other as (x_2, y_2).
    if top_right[0] <= bottom_right[0]:
        (x_1, y_1), (x_2, y_2) = top_right, bottom_right
    else:
        (x_1, y_1), (x_2, y_2) = bottom_right, top_right

    # Left edge x = A's left edge on canvas (0 in A's local coords + tx).
    x_left = tx

    final_corners = [
        (x_1,    y_1),
        (x_1,    y_2),
        (x_left, y_1),
        (x_left, y_2),
    ]

    xs = [c[0] for c in final_corners]
    ys = [c[1] for c in final_corners]
    x0 = max(0,        int(round(min(xs))))
    y0 = max(0,        int(round(min(ys))))
    x1 = min(canvas_w, int(round(max(xs))))
    y1 = min(canvas_h, int(round(max(ys))))
    return x0, y0, x1 - x0, y1 - y0


def build_remap(H, canvas_size):
    """
    Build cv2.remap-style pixel-level (map_x, map_y) tables that warp a
    source image into the canvas via the homography H. Used by the CPU
    path; the GPU path uses build_grid_sample_tensor (in warp.py) on
    the same map_x/map_y arrays.
    """
    W, H_canvas = canvas_size
    H_inv = np.linalg.inv(H)
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H_canvas, dtype=np.float32))
    ones = np.ones_like(xs)
    canvas_coords = np.stack([xs.ravel(), ys.ravel(), ones.ravel()], axis=0)
    src = H_inv @ canvas_coords
    src /= src[2:3, :]
    map_x = src[0].reshape(H_canvas, W).astype(np.float32)
    map_y = src[1].reshape(H_canvas, W).astype(np.float32)
    return map_x, map_y


def build_static_geometry(src_shape_a, src_shape_b, map_ax, map_ay,
                          map_bx, map_by, canvas_size):
    """
    Pre-compute per-camera validity masks, the overlap mask, and the
    overlap bounding box. All per-frame work restricts itself to the
    overlap bbox.

    Returns a dict with:
        only_a_u8       : full-canvas uint8 mask (0/255), pixels only in A
        only_b_u8       : same for B
        overlap_bbox    : (x0, y0, x1, y1) tight bbox around the overlap
        overlap_in_bbox : uint8 mask of the overlap shape inside the bbox
        mask_a_in_bbox  : uint8 mask of A's validity inside the bbox
        mask_b_in_bbox  : same for B
        only_a_in_bbox  : uint8 mask of "only A" inside the bbox
        only_b_in_bbox  : same for B
    """
    W, H = canvas_size
    src_white_a = np.full(src_shape_a[:2], 255, dtype=np.uint8)
    src_white_b = np.full(src_shape_b[:2], 255, dtype=np.uint8)
    mask_a = cv2.remap(src_white_a, map_ax, map_ay, cv2.INTER_NEAREST)
    mask_b = cv2.remap(src_white_b, map_bx, map_by, cv2.INTER_NEAREST)

    overlap_bool = (mask_a > 0) & (mask_b > 0)
    only_a_bool = (mask_a > 0) & ~overlap_bool
    only_b_bool = (mask_b > 0) & ~overlap_bool
    if not overlap_bool.any():
        raise RuntimeError("No overlap between cameras.")

    rows = np.where(overlap_bool.any(axis=1))[0]
    cols = np.where(overlap_bool.any(axis=0))[0]
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1

    overlap_in_bbox = (overlap_bool[y0:y1, x0:x1].astype(np.uint8)) * 255
    mask_a_in_bbox = mask_a[y0:y1, x0:x1].copy()
    mask_b_in_bbox = mask_b[y0:y1, x0:x1].copy()
    only_a_in_bbox = ((only_a_bool[y0:y1, x0:x1]).astype(np.uint8)) * 255
    only_b_in_bbox = ((only_b_bool[y0:y1, x0:x1]).astype(np.uint8)) * 255

    return {
        "only_a_u8": (only_a_bool.astype(np.uint8)) * 255,
        "only_b_u8": (only_b_bool.astype(np.uint8)) * 255,
        "overlap_bbox": (x0, y0, x1, y1),
        "overlap_in_bbox": overlap_in_bbox,
        "mask_a_in_bbox": mask_a_in_bbox,
        "mask_b_in_bbox": mask_b_in_bbox,
        "only_a_in_bbox": only_a_in_bbox,
        "only_b_in_bbox": only_b_in_bbox,
    }
