"""
Depth estimation helper for the depth-aware static-FG selector.

Loads Depth Anything V2 (Small) via Hugging Face transformers on
first call and keeps the pipeline cached in process memory so a
second call in the same run reuses the already-loaded model.

Depth values are normalized to [0, 1] using the 5th/95th percentiles
of the scene's inverse-depth map (clipped). Larger value = closer
to the camera. The normalization is scene-relative, so downstream
thresholds (e.g. stitcher.static_fg.FG_DEPTH_THRESHOLD = 0.4) are
comparable across runs even though the raw model output has no
absolute scale.

Lazy import: `transformers` is only required when this module is
actually used. The rest of the pipeline doesn't need it.
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
            "depth-aware static FG requires `transformers`. Install with:\n"
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

    Call this once the selector is done with depth so the model's
    weights don't sit on the GPU during the main stitching loop.
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
    Return a (H, W) float32 inverse-depth map (larger = closer to
    the camera). Resized to the input frame's resolution so bbox-
    pixel indexing works directly on the result.
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


def normalize_depth(depth_map,
                    percentile_low=5.0,
                    percentile_high=95.0):
    """
    Normalize an inverse-depth map to [0, 1] using the 5th/95th
    percentiles as endpoints (clipped). 1.0 = closest to the camera,
    0.0 = farthest. Returns float32 same shape as input.
    """
    lo = float(np.percentile(depth_map, percentile_low))
    hi = float(np.percentile(depth_map, percentile_high))
    span = max(hi - lo, 1e-6)
    out = (depth_map - lo) / span
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def bbox_median_depth(depth_map_norm, bbox):
    """
    Median normalized depth (in [0, 1], 1 = closest) inside a bbox.
    Returns NaN if the bbox is out of bounds or empty.

    Median (not mean) is used because YOLOE bboxes routinely catch
    a strip of background along the edges; median ignores those
    tails.
    """
    H, W = depth_map_norm.shape
    x1, y1, x2, y2 = (int(round(c)) for c in bbox[:4])
    x1 = max(0, min(W, x1))
    x2 = max(0, min(W, x2))
    y1 = max(0, min(H, y1))
    y2 = max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1:
        return float("nan")
    region = depth_map_norm[y1:y2, x1:x2]
    if region.size == 0:
        return float("nan")
    return float(np.median(region))
