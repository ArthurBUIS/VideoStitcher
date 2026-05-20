"""
Auto-discovery of the static-FG class list for --yoloe_fg_classes.

Orchestrates Steps 0+1+2+3 of the auto-FG-v2 flow on one camera frame:

  Step 0: Qwen2.5-VL inventories the room (color + detail per item)
          (stitcher.vlm_inventory.list_objects_with_details).
          The result is the per-scene YOLOE vocabulary.
  Step 1: YOLOE detects every object from the inventory vocab
          (stitcher.object_inventory.detect_all_objects).
  Step 2: Depth Anything V2 annotates each bbox with a depth value
          in [0, 1] (stitcher.depth_inventory.annotate_inventory_with_depth).
  Step 3: Qwen2.5-VL judges which classes the panorama seam should
          route around (stitcher.vlm_judge.judge_inventory). The
          VLM sees the image AND the depth-annotated inventory and
          ranks by visual importance, not strict FG/BG.

The flow runs BEFORE the stitcher allocates any GPU state, so each
model only lives in VRAM during its own step:

  Qwen (step 0) -> Ollama process, separate VRAM
  YOLOE         -> loaded + released inside detect_all_objects
  Depth         -> loaded by annotate_inventory_with_depth, released
                   by release_depth() before the VLM judge call
  Qwen (step 3) -> Ollama process, separate VRAM

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
from stitcher.vlm_inventory import list_objects_with_details
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
                          inventory_vocab=None,
                          return_details=False):
    """
    Pick a list of static-FG class names for YOLOE to track, by
    chaining the VLM inventory + YOLOE detection + depth annotation
    + VLM judge.

    Args:
        frame_bgr: HxWx3 uint8 BGR image. ONE camera frame is enough
            (the left camera is the usual choice); multi-image input
            is unreliable with Qwen2.5-VL on Ollama.
        yoloe_weights: YOLOE weights path. Default yoloe-11s-seg.pt.
        device: torch device for YOLOE + depth. Default "cuda:0".
        ollama_model: Ollama tag used for BOTH the inventory call
            (step 0) and the judge call (step 3).
        min_confidence: drop YOLOE detections below this score before
            the depth + VLM steps. Default 0.0 keeps everything.
        inventory_vocab: optional pre-baked YOLOE text-class list.
            When provided, Step 0 is skipped and YOLOE looks for
            exactly these classes. Useful for ablation tests and to
            force a known vocabulary (e.g. the hardcoded
            INVENTORY_CLASSES fallback from
            stitcher.object_inventory).
        return_details: if True, also return
            (vocab, records, raw_vlm_text) where vocab is the YOLOE
            text-class list and raw_vlm_text is the judge's raw
            response.

    Returns:
        list of class names to feed --yoloe_fg_classes (e.g.
        ['blue armchair', 'wooden desk', 'dual monitor setup', ...]).
        If return_details is True:
            (classes, vocab, records, raw_vlm_text).

    Raises RuntimeError with an actionable hint if YOLOE finds
    nothing, if Ollama is unreachable, or if the VLM returns no
    usable names at any step.
    """
    # Step 0: VLM inventory (skipped when caller supplies a vocab).
    if inventory_vocab is None:
        vocab = list_objects_with_details(
            frame_bgr, model_name=ollama_model,
        )
    else:
        vocab = list(inventory_vocab)
    if not vocab:
        raise RuntimeError(
            "auto-FG-v2: inventory vocabulary is empty. The VLM "
            "produced no phrases (or an empty inventory_vocab was "
            "passed in)."
        )

    # Step 1
    detections = detect_all_objects(
        frame_bgr,
        yoloe_weights=yoloe_weights,
        device=device,
        class_list=vocab,
        min_confidence=min_confidence,
    )
    if not detections:
        raise RuntimeError(
            "auto-FG-v2: YOLOE detected no objects in the frame "
            "with the inventory vocab. Try a different frame, "
            "lower --min_confidence, or relax the inventory prompt."
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
        return classes, vocab, records, raw
    return classes
