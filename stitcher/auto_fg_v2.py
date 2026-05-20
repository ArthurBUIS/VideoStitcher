"""
Auto-discovery of the static-FG class list for --yoloe_fg_classes.

Orchestrates Steps 1+2+3 of the auto-FG-v2 flow on one camera frame:

  Step 1: YOLOE detects every object in a broad indoor vocabulary
          (stitcher.object_inventory.detect_all_objects).
  Step 2: Depth Anything V2 annotates each bbox with a depth value
          in [0, 1] (stitcher.depth_inventory.annotate_inventory_with_depth).
  Step 3: Qwen2.5-VL judges which classes the panorama seam should
          route around (stitcher.vlm_judge.judge_inventory). The
          VLM sees the image AND the depth-annotated inventory and
          ranks by visual importance, not strict FG/BG.

The flow runs BEFORE the stitcher allocates any GPU state, so each
of YOLOE / Depth Anything / (Ollama-hosted) Qwen lives in VRAM only
during its own step:

  YOLOE         -> loaded + released inside detect_all_objects
  Depth         -> loaded by annotate_inventory_with_depth, released
                   by release_depth() before the VLM call
  Qwen2.5-VL    -> runs in a separate Ollama process, doesn't share
                   our Python process's VRAM

Two entry points:
  * tools/suggest_fg_classes_v2.py -- standalone CLI; prints the
    comma-separated kept-class list to stdout.
  * --yoloe_fg_classes auto -- video_stitcher_seam_gpu.py picks up
    this sentinel in main() and runs suggest_fg_classes_v2() before
    booting the stitching pipeline.
"""

import cv2

from stitcher.depth_inventory import (
    annotate_inventory_with_depth,
    release_depth,
)
from stitcher.object_inventory import detect_all_objects
from stitcher.vlm_judge import DEFAULT_OLLAMA_MODEL, judge_inventory


# Sentinels for --yoloe_fg_classes that trigger auto-discovery.
AUTO_SENTINELS = ("auto", "automatic")


def is_auto_sentinel(values):
    """
    True if `values` is the auto-discovery sentinel (exactly one item
    matching AUTO_SENTINELS, case-insensitive). Anything else (a
    literal class list, an empty list) returns False.
    """
    if not values or len(values) != 1:
        return False
    return str(values[0]).strip().lower() in AUTO_SENTINELS


def read_frame_zero(video_path):
    """Read frame 0 of a video file as BGR uint8."""
    cap = cv2.VideoCapture(video_path)
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(
            f"Could not read frame 0 from {video_path}"
        )
    return frame


def suggest_fg_classes_v2(frame_bgr,
                          yoloe_weights="yoloe-11s-seg.pt",
                          device="cuda:0",
                          ollama_model=DEFAULT_OLLAMA_MODEL,
                          min_confidence=0.0,
                          return_details=False):
    """
    Pick a list of static-FG class names for YOLOE to track, by
    chaining the object inventory + depth annotation + VLM judge.

    Args:
        frame_bgr: HxWx3 uint8 BGR image. ONE camera frame is enough
            (the left camera is the usual choice); multi-image input
            is unreliable with Qwen2.5-VL on Ollama.
        yoloe_weights: YOLOE weights path. Default yoloe-11s-seg.pt.
        device: torch device for YOLOE + depth. Default "cuda:0".
        ollama_model: Ollama tag for the VLM judge.
        min_confidence: drop YOLOE detections below this score before
            the depth + VLM steps. Default 0.0 keeps everything.
        return_details: if True, also return (records, raw_vlm_text)
            -- useful when the caller wants to log / save them.

    Returns:
        list of class names to feed --yoloe_fg_classes (e.g.
        ['chair', 'tv', 'desk', 'houseplant']).
        If return_details is True: (classes, records, raw_vlm_text).

    Raises RuntimeError with an actionable hint if YOLOE finds
    nothing, if Ollama is unreachable, or if the VLM returns no
    usable names.
    """
    # Step 1
    detections = detect_all_objects(
        frame_bgr,
        yoloe_weights=yoloe_weights,
        device=device,
        min_confidence=min_confidence,
    )
    if not detections:
        raise RuntimeError(
            "auto-FG-v2: YOLOE detected no objects in the frame. "
            "Try a different frame, lower --min_confidence, or "
            "extend INVENTORY_CLASSES."
        )

    # Step 2
    records = annotate_inventory_with_depth(frame_bgr, detections)
    if not records:
        raise RuntimeError(
            "auto-FG-v2: depth annotation produced no records "
            "(every bbox was degenerate)."
        )
    # Free the depth pipeline before the VLM call so we don't sit on
    # its VRAM during inference (Ollama runs in its own process, but
    # depth's transformers pipeline stays in ours otherwise).
    release_depth()

    # Step 3
    classes, raw = judge_inventory(
        frame_bgr, records,
        model_name=ollama_model,
        return_raw=True,
    )

    if return_details:
        return classes, records, raw
    return classes
