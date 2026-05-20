"""
Annotate the YOLOE object inventory with a per-bbox depth estimate.

Step B of the auto-FG-v2 discovery flow: takes the bbox dict from
stitcher.object_inventory.detect_all_objects and decorates each
detection with a depth value, so the downstream VLM judge can see
how close each object is to the camera.

Depth comes from Depth Anything V2 (Small) via HuggingFace
transformers, lazy-loaded and process-cached so a discovery flow
that calls in multiple times pays the load cost only once.

Depth values are normalized to [0, 1] using the 5th / 95th
percentiles of the scene's inverse-depth map (clipped). Larger =
closer to the camera. The normalization is scene-relative, so the
VLM sees a comparable scalar across runs even though the raw model
output has no absolute scale.
"""

import gc

import cv2
import numpy as np


_DEPTH_PIPE = None  # process-cached HF pipeline


def _get_depth_pipe():
    """Lazy-load the Depth Anything V2 (Small) HF pipeline."""
    global _DEPTH_PIPE
    if _DEPTH_PIPE is not None:
        return _DEPTH_PIPE
    try:
        from transformers import pipeline
    except ImportError as e:
        raise RuntimeError(
            "depth annotation requires `transformers`. Install with:\n"
            "    pip install transformers"
        ) from e
    _DEPTH_PIPE = pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
    )
    return _DEPTH_PIPE


def release_depth():
    """
    Drop the cached depth pipeline + free GPU memory.

    Call this after the discovery flow is done with depth (e.g. just
    before the VLM judge step) so the depth model's weights don't sit
    on the GPU forever.
    """
    global _DEPTH_PIPE
    _DEPTH_PIPE = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def estimate_depth(frame_bgr):
    """
    Return a (H, W) float32 inverse-depth map (larger = closer).

    Resized to the input frame's resolution so bbox-pixel indexing
    works directly on the result.
    """
    from PIL import Image

    pipe = _get_depth_pipe()
    H, W = frame_bgr.shape[:2]
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    result = pipe(img)
    depth_pil = result["depth"]
    depth = np.array(depth_pil, dtype=np.float32)
    if depth.shape != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
    return depth


def _bbox_median_depth(depth_map, bbox):
    """Median inverse-depth inside a bbox (clipped to image bounds)."""
    H, W = depth_map.shape
    x1, y1, x2, y2 = (int(round(c)) for c in bbox[:4])
    x1 = max(0, min(W, x1))
    x2 = max(0, min(W, x2))
    y1 = max(0, min(H, y1))
    y2 = max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1:
        return float("nan")
    region = depth_map[y1:y2, x1:x2]
    if region.size == 0:
        return float("nan")
    return float(np.median(region))


def annotate_inventory_with_depth(frame_bgr, detections,
                                  percentile_low=5.0,
                                  percentile_high=95.0,
                                  depth_map=None):
    """
    Annotate every detection with a per-bbox depth value.

    Args:
        frame_bgr: HxWx3 uint8 BGR image (the same frame passed to
            detect_all_objects).
        detections: dict {class_name: [(x1, y1, x2, y2, conf), ...]}
            as returned by stitcher.object_inventory.detect_all_objects.
        percentile_low/high: percentiles of the inverse-depth map
            used as the 0/1 endpoints for normalization. Clipping to
            (5, 95) is robust to outliers (sky pixels, glints, ...).
        depth_map: optional precomputed (H, W) inverse-depth map. If
            None, estimate_depth(frame_bgr) is called. Provided so
            higher-level code can reuse a depth map it already has.

    Returns: list of dicts, one per detection:
        [{"class": str,
          "bbox": (x1, y1, x2, y2),
          "conf": float,
          "depth": float},   # 0..1, 1 = closest to camera
         ...]
    Sorted by depth descending (closest first) -- the natural order
    for the VLM judge to read.

    Median (not mean) is used for the bbox-depth reduction because
    YOLOE bboxes routinely include background pixels along the edges
    (e.g. a chair bbox that catches some floor + wall). Median ignores
    those tails.
    """
    if depth_map is None:
        depth_map = estimate_depth(frame_bgr)

    lo = float(np.percentile(depth_map, percentile_low))
    hi = float(np.percentile(depth_map, percentile_high))
    span = max(hi - lo, 1e-6)

    out = []
    for name, boxes in detections.items():
        for (x1, y1, x2, y2, conf) in boxes:
            raw = _bbox_median_depth(depth_map, (x1, y1, x2, y2))
            if np.isnan(raw):
                continue
            depth = (raw - lo) / span
            depth = float(np.clip(depth, 0.0, 1.0))
            out.append({
                "class": name,
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "conf": float(conf),
                "depth": depth,
            })

    out.sort(key=lambda r: r["depth"], reverse=True)
    return out
