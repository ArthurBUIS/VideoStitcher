"""
Homography, canvas, remap tables, static per-camera geometry, autocrop.

Everything here runs once at startup. The outputs are reused unchanged
for every frame.

Both 2-camera and 3-camera setups are supported. The 3-camera variants
use Center as the geometric anchor: H_left_to_center and
H_right_to_center are estimated independently, then both warped into a
single canvas. Per-frame work is split into two OverlapContexts (L<>C
and C<>R) that each carry the same shape of data the 2-camera path
used to carry as a single dict.
"""

from dataclasses import dataclass
from typing import Any, Dict

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
    Largest axis-aligned rectangle inscribed inside the union polygon of
    camera A and warped camera B on the canvas, maximising the x-extent.

    Algorithm:
      1. Rasterise the union of A's footprint and warped B's footprint
         on a canvas-sized mask, then extract its outer boundary as a
         polygon via cv2.findContours + approxPolyDP. The polygon's
         vertices are the genuine corners of the union — input corners
         that lie INSIDE the other input do not appear, while edge-edge
         intersection points DO. This is strictly more informative than
         using only the 8 input corners, because interior input corners
         can carry extreme y values that would shrink the strip in step
         3 unnecessarily.
      2. Classify each polygon vertex as "top" or "bottom" by comparing
         its y to A's horizontal midline (y = ty + h_a/2). Above the
         midline (smaller y) = top, below = bottom. Using A's midline
         (A being the unwarped reference) is more robust than a polygon
         diagonal: it handles edge-edge intersection points and
         asymmetric polygon shapes uniformly.
      3. Define the y-strip:
             y_top    = max(y of top-set vertices)
             y_bottom = min(y of bottom-set vertices)
         Above y_top the polygon's top edge starts to bend; below
         y_bottom the bottom edge does. Inside [y_top, y_bottom] the
         polygon is a right trapezoid (or rectangle).
      4. Inscribe the rectangle by collapsing any slanted edges:
             rect.left  = max(polygon.left  at y_top, polygon.left  at y_bottom)
             rect.right = min(polygon.right at y_top, polygon.right at y_bottom)
         where polygon.{left,right}(y) are the smallest and largest x at
         which the polygon contains row y. Min/max over the strip happens
         at one of the endpoints because each polygon edge is straight.

    Returns (x, y, w, h) in canvas coords. Falls back to the full canvas
    if A and warped B are disjoint or the strip is degenerate.
    """
    canvas_w, canvas_h = canvas_size
    h_a, w_a = shape_a[:2]
    h_b, w_b = shape_b[:2]
    tx, ty = float(T[0, 2]), float(T[1, 2])

    # A's 4 corners on canvas (axis-aligned).
    A_corners = np.float32([
        [tx,         ty       ],
        [tx + w_a,   ty       ],
        [tx + w_a,   ty + h_a ],
        [tx,         ty + h_a ],
    ])

    # B's 4 corners on canvas (warped via H_b_to_a + T).
    b_src = np.float32([[0, 0], [w_b, 0], [w_b, h_b], [0, h_b]]).reshape(-1, 1, 2)
    b_canvas = cv2.perspectiveTransform(b_src, H_b_to_a).reshape(-1, 2)
    b_canvas += np.array([tx, ty], dtype=np.float32)

    # ---- Step 1: rasterise the union, recover its outer polygon ----------
    mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    cv2.fillPoly(mask, [A_corners.astype(np.int32)], 255)
    cv2.fillPoly(mask, [b_canvas .astype(np.int32)], 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0, 0, canvas_w, canvas_h
    # If A and warped B happen not to overlap there will be two contours;
    # the larger one is the more useful side (autocrop has to fit inside
    # one connected region).
    contour = max(contours, key=cv2.contourArea)
    # approxPolyDP collapses the pixel-jagged boundary down to the genuine
    # polygon corners. 2 px epsilon is generous at typical canvas sizes.
    polygon = cv2.approxPolyDP(contour, epsilon=2.0, closed=True) \
                 .reshape(-1, 2).astype(np.float64)
    if len(polygon) < 3:
        return 0, 0, canvas_w, canvas_h

    # ---- Step 2: classify each vertex as top / bottom --------------------
    # Use A's horizontal midline as the separator: y_mid = ty + h_a/2.
    # A vertex is "top" if its y is above the midline (smaller y in image
    # coords) and "bottom" otherwise. This works on every polygon vertex
    # uniformly — input corners as well as edge-edge intersection points,
    # which a polygon-diagonal heuristic mishandles when the polygon is
    # asymmetric.
    y_mid = ty + h_a / 2.0

    top_ys    = []
    bottom_ys = []
    for v in polygon:
        if v[1] < y_mid:
            top_ys.append(v[1])
        else:
            bottom_ys.append(v[1])

    # ---- Step 3: y-strip where the polygon is rectangular in y -----------
    if not top_ys or not bottom_ys:
        # Polygon entirely above or below A's midline — degenerate.
        return 0, 0, canvas_w, canvas_h
    y_top    = max(top_ys)
    y_bottom = min(bottom_ys)
    if y_top >= y_bottom:
        return 0, 0, canvas_w, canvas_h

    # ---- Step 4: polygon x-range at the two strip boundaries -------------
    def _polygon_x_range_at_y(poly, y):
        """(min_x, max_x) at row y by intersecting y with every polygon
        edge. For any simple polygon this gives the leftmost/rightmost
        pixel coordinate that lies inside the polygon at that row."""
        crossings = []
        n = len(poly)
        for i in range(n):
            p1, p2 = poly[i], poly[(i + 1) % n]
            y1, y2 = p1[1], p2[1]
            if (y1 <= y <= y2) or (y2 <= y <= y1):
                if abs(y2 - y1) < 1e-9:
                    crossings.append(p1[0])
                    crossings.append(p2[0])
                else:
                    t = (y - y1) / (y2 - y1)
                    crossings.append(p1[0] + t * (p2[0] - p1[0]))
        return (min(crossings), max(crossings)) if crossings else (None, None)

    poly_l_top, poly_r_top = _polygon_x_range_at_y(polygon, y_top)
    poly_l_bot, poly_r_bot = _polygon_x_range_at_y(polygon, y_bottom)
    if None in (poly_l_top, poly_r_top, poly_l_bot, poly_r_bot):
        return 0, 0, canvas_w, canvas_h

    rect_left  = max(poly_l_top, poly_l_bot)
    rect_right = min(poly_r_top, poly_r_bot)

    x0 = max(0,        int(np.ceil (rect_left )))
    y0 = max(0,        int(np.ceil (y_top     )))
    x1 = min(canvas_w, int(np.floor(rect_right)))
    y1 = min(canvas_h, int(np.floor(y_bottom  )))

    if x1 <= x0 or y1 <= y0:
        return 0, 0, canvas_w, canvas_h
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


# ---------------------------------------------------------------------------
# 3-camera primitives
# ---------------------------------------------------------------------------
#
# These extend the 2-camera path above. The 2-camera entry points
# (estimate_homography, build_remap, build_static_geometry, etc.)
# are unchanged and remain in use by the file-mode 2-cam pipeline.
#
# Naming convention:
#   L = Left, C = Center, R = Right
#   H_L_to_C   maps Left's pixel coords into Center's frame
#   H_R_to_C   maps Right's pixel coords into Center's frame
#   *_in_bbox  arrays are cropped to the relevant OverlapContext bbox
#
# Center is the geometric anchor: the canvas translation T is computed
# so Center's top-left corner ends up at (0, 0) before autocrop, same
# rule the 2-camera path uses for its "A" camera.


@dataclass
class OverlapContext:
    """
    All per-overlap geometry data the per-frame pipeline needs.

    The 2-camera pipeline has one such context implicitly (the dict
    returned by build_static_geometry). The 3-camera pipeline has two,
    one for each adjacent pair: L<>C and C<>R. Each context's fields
    have the same shapes as the 2-camera dict's, so per-frame code
    that operates on "the overlap" can be reused unchanged by feeding
    it one context at a time.

    bbox        : (x0, y0, x1, y1) tight bbox around this overlap on
                  the canvas. All other arrays in this context are
                  cropped to bbox-shape: (bbox.h, bbox.w).
    overlap     : uint8 mask of the overlap shape inside bbox.
    mask_left   : uint8 mask of the LEFT camera's validity inside bbox.
                  ("left" here means whichever camera is geometrically
                  to the LEFT of the seam in this overlap, NOT the
                  literal Left camera -- for the L<>C overlap this is
                  L; for the C<>R overlap this is C.)
    mask_right  : same for the geometrically-right camera in this
                  overlap. For L<>C this is C; for C<>R this is R.
    only_left   : "left and not in overlap" cropped to bbox. Drives
                  the build_soft_mask_fast "force alpha to 1" fixup.
    only_right  : same for the right side. Drives the "force alpha
                  to 0" fixup.
    """

    bbox: tuple
    overlap: np.ndarray
    mask_left: np.ndarray
    mask_right: np.ndarray
    only_left: np.ndarray
    only_right: np.ndarray

    @property
    def shape(self) -> tuple:
        """(H_bb, W_bb) -- bbox height, bbox width."""
        x0, y0, x1, y1 = self.bbox
        return (y1 - y0, x1 - x0)

    def as_static_dict(self) -> Dict[str, Any]:
        """
        Return a dict whose keys match what the 2-camera per-frame
        code reads off `static` (build_static_geometry's return).
        Lets us pass an OverlapContext into compute_cost_and_ema_gpu
        / build_soft_mask_fast / etc. with no API change on those
        functions.
        """
        return {
            "overlap_bbox": self.bbox,
            "overlap_in_bbox": self.overlap,
            "mask_a_in_bbox": self.mask_left,
            "mask_b_in_bbox": self.mask_right,
            "only_a_in_bbox": self.only_left,
            "only_b_in_bbox": self.only_right,
        }


def compute_canvas_3cam(shape_L, shape_C, shape_R,
                        H_L_to_C, H_R_to_C):
    """
    Compute the canvas size + Center-anchored translation for a
    3-camera panorama.

    Returns (canvas_size, T, H_L_to_canvas, H_C_to_canvas, H_R_to_canvas)
    where:
        canvas_size       : (width, height)
        T                 : 3x3 translation that brings Center's
                            top-left to canvas (0, 0)
        H_*_to_canvas     : T composed with each camera's homography
                            into Center's frame (Center's own homography
                            into Center's frame is the identity, so
                            H_C_to_canvas = T)

    Naming parallel to compute_canvas: Center plays the role of A
    (the unwarped anchor); Left and Right play the role of B (warped
    inputs).
    """
    h_L, w_L = shape_L[:2]
    h_C, w_C = shape_C[:2]
    h_R, w_R = shape_R[:2]

    # Center's corners are axis-aligned in Center's frame.
    corners_C = np.float32(
        [[0, 0], [w_C, 0], [w_C, h_C], [0, h_C]]
    ).reshape(-1, 1, 2)
    # Left and Right corners get warped into Center's frame first.
    corners_L_in_C = cv2.perspectiveTransform(
        np.float32([[0, 0], [w_L, 0], [w_L, h_L], [0, h_L]]).reshape(-1, 1, 2),
        H_L_to_C,
    )
    corners_R_in_C = cv2.perspectiveTransform(
        np.float32([[0, 0], [w_R, 0], [w_R, h_R], [0, h_R]]).reshape(-1, 1, 2),
        H_R_to_C,
    )
    all_corners = np.concatenate(
        [corners_C, corners_L_in_C, corners_R_in_C], axis=0
    )
    x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)
    tx, ty = -x_min, -y_min
    T = np.array(
        [[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64
    )
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    H_C_to_canvas = T.copy()
    H_L_to_canvas = T @ H_L_to_C
    H_R_to_canvas = T @ H_R_to_C
    return (
        (canvas_w, canvas_h),
        T,
        H_L_to_canvas,
        H_C_to_canvas,
        H_R_to_canvas,
    )


def _bbox_from_overlap_bool(overlap_bool):
    """Tight bbox around a boolean overlap mask. Returns
    (x0, y0, x1, y1) or None if the mask is empty."""
    if not overlap_bool.any():
        return None
    rows = np.where(overlap_bool.any(axis=1))[0]
    cols = np.where(overlap_bool.any(axis=0))[0]
    return (
        int(cols[0]),
        int(rows[0]),
        int(cols[-1]) + 1,
        int(rows[-1]) + 1,
    )


def build_static_geometry_3cam(src_shape_L, src_shape_C, src_shape_R,
                                map_Lx, map_Ly,
                                map_Cx, map_Cy,
                                map_Rx, map_Ry,
                                canvas_size):
    """
    3-camera analogue of build_static_geometry: produces TWO
    OverlapContexts, one for the L<>C overlap and one for C<>R, plus
    canvas-level per-camera validity masks used by the compositor for
    the "only this camera sees this pixel" regions.

    Returns a dict with:
        overlap_LC          : OverlapContext for the Left<>Center overlap
        overlap_CR          : OverlapContext for the Center<>Right overlap
        only_L_u8           : full-canvas uint8 mask, pixels only in Left
        only_C_u8           : same for Center
        only_R_u8           : same for Right
        x_mid               : x coord on canvas of Center's projected midline
                              -- the seam-cap reference (canvas pixels).

    Raises RuntimeError if either of the two overlaps is empty
    (degenerate camera geometry).
    """
    W, H = canvas_size

    src_white_L = np.full(src_shape_L[:2], 255, dtype=np.uint8)
    src_white_C = np.full(src_shape_C[:2], 255, dtype=np.uint8)
    src_white_R = np.full(src_shape_R[:2], 255, dtype=np.uint8)
    mask_L = cv2.remap(src_white_L, map_Lx, map_Ly, cv2.INTER_NEAREST)
    mask_C = cv2.remap(src_white_C, map_Cx, map_Cy, cv2.INTER_NEAREST)
    mask_R = cv2.remap(src_white_R, map_Rx, map_Ry, cv2.INTER_NEAREST)

    L_bool = mask_L > 0
    C_bool = mask_C > 0
    R_bool = mask_R > 0

    overlap_LC_bool = L_bool & C_bool
    overlap_CR_bool = C_bool & R_bool

    if not overlap_LC_bool.any():
        raise RuntimeError("No overlap between Left and Center cameras.")
    if not overlap_CR_bool.any():
        raise RuntimeError("No overlap between Center and Right cameras.")

    # "only X" canvas-level masks: pixel is visible to X and no other
    # camera. With 3 cameras this includes the triple-overlap region
    # not as "only" anyone, which is correct -- inside L<>C<>R triple
    # the seam-cap decision rule says C dominates the middle band and
    # the seams stay within their respective halves anyway.
    only_L_bool = L_bool & ~C_bool & ~R_bool
    only_C_bool = C_bool & ~L_bool & ~R_bool
    only_R_bool = R_bool & ~C_bool & ~L_bool

    # ---- L<>C overlap context ------------------------------------------
    bbox_LC = _bbox_from_overlap_bool(overlap_LC_bool)
    x0, y0, x1, y1 = bbox_LC
    overlap_LC = (overlap_LC_bool[y0:y1, x0:x1].astype(np.uint8)) * 255
    mask_L_in_LC = mask_L[y0:y1, x0:x1].copy()
    mask_C_in_LC = mask_C[y0:y1, x0:x1].copy()
    # "only-left" for this overlap means "only L sees, not C" within
    # the bbox -- drives the soft mask "force alpha to 1" rule for
    # the Left side. Symmetric "only_right" = "only C, not L" inside
    # the bbox forces alpha to 0 (full Center) on its side.
    only_L_in_LC_bool = mask_L[y0:y1, x0:x1] > 0
    only_L_in_LC_bool &= ~overlap_LC_bool[y0:y1, x0:x1]
    only_C_in_LC_bool = mask_C[y0:y1, x0:x1] > 0
    only_C_in_LC_bool &= ~overlap_LC_bool[y0:y1, x0:x1]

    ctx_LC = OverlapContext(
        bbox=bbox_LC,
        overlap=overlap_LC,
        mask_left=mask_L_in_LC,
        mask_right=mask_C_in_LC,
        only_left=(only_L_in_LC_bool.astype(np.uint8)) * 255,
        only_right=(only_C_in_LC_bool.astype(np.uint8)) * 255,
    )

    # ---- C<>R overlap context ------------------------------------------
    bbox_CR = _bbox_from_overlap_bool(overlap_CR_bool)
    x0, y0, x1, y1 = bbox_CR
    overlap_CR = (overlap_CR_bool[y0:y1, x0:x1].astype(np.uint8)) * 255
    mask_C_in_CR = mask_C[y0:y1, x0:x1].copy()
    mask_R_in_CR = mask_R[y0:y1, x0:x1].copy()
    only_C_in_CR_bool = mask_C[y0:y1, x0:x1] > 0
    only_C_in_CR_bool &= ~overlap_CR_bool[y0:y1, x0:x1]
    only_R_in_CR_bool = mask_R[y0:y1, x0:x1] > 0
    only_R_in_CR_bool &= ~overlap_CR_bool[y0:y1, x0:x1]

    ctx_CR = OverlapContext(
        bbox=bbox_CR,
        overlap=overlap_CR,
        mask_left=mask_C_in_CR,
        mask_right=mask_R_in_CR,
        only_left=(only_C_in_CR_bool.astype(np.uint8)) * 255,
        only_right=(only_R_in_CR_bool.astype(np.uint8)) * 255,
    )

    # ---- x_mid: Center's projected mid-x on canvas ---------------------
    # Per the design discussion: seam_LC stays in (canvas x <= x_mid)
    # and seam_CR stays in (canvas x >= x_mid). Defining x_mid as the
    # midpoint of Center's projected footprint on the canvas is
    # straightforward because Center is the anchor: its footprint is
    # an axis-aligned rectangle of width w_C starting at (tx, ty).
    # We recover tx from where Center's mask actually starts on canvas.
    cols_C = np.where(C_bool.any(axis=0))[0]
    if cols_C.size == 0:
        # Degenerate, but fall back to canvas centre rather than
        # crashing -- the bbox checks above should have caught it.
        x_mid = W // 2
    else:
        x_mid = int((cols_C[0] + cols_C[-1] + 1) // 2)

    return {
        "overlap_LC": ctx_LC,
        "overlap_CR": ctx_CR,
        "only_L_u8": (only_L_bool.astype(np.uint8)) * 255,
        "only_C_u8": (only_C_bool.astype(np.uint8)) * 255,
        "only_R_u8": (only_R_bool.astype(np.uint8)) * 255,
        "x_mid": x_mid,
    }


def find_autocrop_rect_3cam(H_L_to_C, H_R_to_C,
                             shape_L, shape_C, shape_R,
                             canvas_size, T):
    """
    3-camera analogue of find_autocrop_rect: the largest axis-aligned
    rectangle inscribed in the UNION of the three camera footprints on
    canvas.

    Implementation: identical to the 2-camera version, just with three
    polygons rasterised into the union mask instead of two. The y-strip
    + polygon-x-range logic downstream is camera-count-agnostic.
    """
    canvas_w, canvas_h = canvas_size
    h_L, w_L = shape_L[:2]
    h_C, w_C = shape_C[:2]
    h_R, w_R = shape_R[:2]
    tx, ty = float(T[0, 2]), float(T[1, 2])

    # Center's corners, axis-aligned on canvas.
    C_corners = np.float32([
        [tx,         ty       ],
        [tx + w_C,   ty       ],
        [tx + w_C,   ty + h_C ],
        [tx,         ty + h_C ],
    ])

    # Left and Right corners, warped via their respective H_*_to_C + T.
    L_src = np.float32(
        [[0, 0], [w_L, 0], [w_L, h_L], [0, h_L]]
    ).reshape(-1, 1, 2)
    L_canvas = cv2.perspectiveTransform(L_src, H_L_to_C).reshape(-1, 2)
    L_canvas += np.array([tx, ty], dtype=np.float32)
    R_src = np.float32(
        [[0, 0], [w_R, 0], [w_R, h_R], [0, h_R]]
    ).reshape(-1, 1, 2)
    R_canvas = cv2.perspectiveTransform(R_src, H_R_to_C).reshape(-1, 2)
    R_canvas += np.array([tx, ty], dtype=np.float32)

    mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    cv2.fillPoly(mask, [C_corners.astype(np.int32)], 255)
    cv2.fillPoly(mask, [L_canvas .astype(np.int32)], 255)
    cv2.fillPoly(mask, [R_canvas .astype(np.int32)], 255)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return 0, 0, canvas_w, canvas_h
    contour = max(contours, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(
        contour, epsilon=2.0, closed=True
    ).reshape(-1, 2).astype(np.float64)
    if len(polygon) < 3:
        return 0, 0, canvas_w, canvas_h

    y_mid = ty + h_C / 2.0
    top_ys = []
    bottom_ys = []
    for v in polygon:
        if v[1] < y_mid:
            top_ys.append(v[1])
        else:
            bottom_ys.append(v[1])
    if not top_ys or not bottom_ys:
        return 0, 0, canvas_w, canvas_h
    y_top = max(top_ys)
    y_bottom = min(bottom_ys)
    if y_top >= y_bottom:
        return 0, 0, canvas_w, canvas_h

    def _polygon_x_range_at_y(poly, y):
        crossings = []
        n = len(poly)
        for i in range(n):
            p1, p2 = poly[i], poly[(i + 1) % n]
            y1, y2 = p1[1], p2[1]
            if (y1 <= y <= y2) or (y2 <= y <= y1):
                if abs(y2 - y1) < 1e-9:
                    crossings.append(p1[0])
                    crossings.append(p2[0])
                else:
                    t = (y - y1) / (y2 - y1)
                    crossings.append(p1[0] + t * (p2[0] - p1[0]))
        return (min(crossings), max(crossings)) if crossings else (None, None)

    poly_l_top, poly_r_top = _polygon_x_range_at_y(polygon, y_top)
    poly_l_bot, poly_r_bot = _polygon_x_range_at_y(polygon, y_bottom)
    if None in (poly_l_top, poly_r_top, poly_l_bot, poly_r_bot):
        return 0, 0, canvas_w, canvas_h

    rect_left = max(poly_l_top, poly_l_bot)
    rect_right = min(poly_r_top, poly_r_bot)
    x0 = max(0,        int(np.ceil (rect_left )))
    y0 = max(0,        int(np.ceil (y_top     )))
    x1 = min(canvas_w, int(np.floor(rect_right)))
    y1 = min(canvas_h, int(np.floor(y_bottom  )))
    if x1 <= x0 or y1 <= y0:
        return 0, 0, canvas_w, canvas_h
    return x0, y0, x1 - x0, y1 - y0
