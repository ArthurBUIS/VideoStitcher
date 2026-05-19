"""
Depth estimation helper used by the depth-aware variant of the
auto-FG discovery flow.

Loads Depth Anything V2 (Small) via Hugging Face transformers on
first call and keeps the pipeline cached in process memory so the
second frame of a discovery run reuses the already-loaded model.

Lazy import: `transformers` is only required when depth filtering
is actually used. The rest of the pipeline doesn't need it.

Output is INVERSE depth (larger value = closer to the camera) at
the same (H, W) shape as the input frame, so subsequent bbox-
median lookups index pixel coords directly.
"""

import cv2
import numpy as np


_DEPTH_PIPE = None  # process-cached HF pipeline


def _get_depth_pipe():
    """Lazy-load the Depth Anything V2 (Small) pipeline."""
    global _DEPTH_PIPE
    if _DEPTH_PIPE is not None:
        return _DEPTH_PIPE
    try:
        from transformers import pipeline
    except ImportError as e:
        raise RuntimeError(
            "depth-aware --depth_threshold requires the "
            "`transformers` Python package. Install with:\n"
            "    pip install transformers"
        ) from e
    _DEPTH_PIPE = pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
    )
    return _DEPTH_PIPE


def estimate_depth(frame_bgr):
    """
    Return a (H, W) float32 inverse-depth map for an OpenCV BGR
    frame. Larger value = closer to the camera.

    Depth Anything V2 internally accepts a PIL RGB image and returns
    a result dict containing the depth as a PIL image. We convert to
    numpy and resize to the input frame's resolution if the model
    happened to emit a downsampled tensor, so a (y, x) bbox lookup
    indexes the right pixel.

    First call is slow (~2-5 s model load on a T1000); subsequent
    calls in the same process are ~100-300 ms on 1080p frames.
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


# ---------------------------------------------------------------------------
# Classification helpers (pure functions; no model / network I/O)
# ---------------------------------------------------------------------------

def classify_box_label(depth_map, bbox, scene_reference, threshold):
    """
    Classify a single bbox as 'fg' or 'bg' from its bbox-median depth.

    Args:
        depth_map: (H, W) float; larger value = closer to the camera
            (Depth Anything V2 output).
        bbox: (x1, y1, x2, y2[, ...]) in image pixel coords. Extra
            tuple elements (e.g. a confidence) are ignored.
        scene_reference: scalar baseline; typically a percentile of
            the depth map (median is a reasonable default).
        threshold: bbox is 'fg' if median(depth[bbox]) >
            scene_reference * threshold. threshold = 1.0 means
            "bbox closer than scene median". Higher threshold =
            stricter foreground criterion (fewer 'fg' labels).

    Returns: 'fg' or 'bg'. Out-of-bounds or degenerate bboxes
    default to 'bg' (the safer answer for the seam-avoidance use
    case: if we can't tell, don't flag it).
    """
    H, W = depth_map.shape
    x1, y1, x2, y2 = (int(round(c)) for c in bbox[:4])
    x1 = max(0, min(W, x1))
    x2 = max(0, min(W, x2))
    y1 = max(0, min(H, y1))
    y2 = max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1:
        return "bg"
    region = depth_map[y1:y2, x1:x2]
    if region.size == 0:
        return "bg"
    return ("fg" if float(np.median(region)) > scene_reference * threshold
            else "bg")


def classify_classes_by_depth(boxes_by_class, depth_map, threshold,
                              reference_percentile=50.0):
    """
    Given per-class bboxes and a depth map, return a dict
    {class_id: 'fg' | 'bg'}.

    A class is labelled 'fg' if ANY of its detections is classified
    as foreground -- one foreground sighting is enough to keep the
    class in the candidate set.

    Classes with no detections at all are OMITTED from the output
    dict (not 'bg'). Callers that want to distinguish "detected but
    background" from "not detected at all" can check key membership.

    Args:
        boxes_by_class: dict {class_id: [(x1,y1,x2,y2,...), ...]}.
            Matches the shape of PersonSegmenter.predict_classes_boxes.
        depth_map: (H, W) float inverse-depth map.
        threshold: see classify_box_label.
        reference_percentile: percentile of the depth map used as
            scene reference (50 = median; 60-75 leans the reference
            further back, marking more bboxes as 'fg').
    """
    if not boxes_by_class:
        return {}
    scene_ref = float(np.percentile(depth_map, reference_percentile))
    out = {}
    for cid, boxes in boxes_by_class.items():
        if not boxes:
            continue
        labels = [classify_box_label(depth_map, b, scene_ref, threshold)
                  for b in boxes]
        out[cid] = "fg" if "fg" in labels else "bg"
    return out
