"""
Auto-discovery of the static-FG class list for --yoloe_fg_classes.

The discovery flow itself runs OFF the main pipeline now: a standalone
CLI (tools/suggest_fg_classes_v2.py) chains the four steps below and
writes a JSON file. The main video pipeline, when given
`--yoloe_fg_classes auto`, just reads that JSON.

  Step 0: Qwen2.5-VL inventories the room (color + detail per item)
          (stitcher.vlm_inventory.list_objects_with_details).
          The result is the per-scene YOLOE vocabulary.
  Step 1: YOLOE detects every object from the inventory vocab
          (stitcher.object_inventory.detect_all_objects).
  Step 2: Depth Anything V2 annotates each bbox with a depth value
          in [0, 1] (stitcher.depth_inventory.annotate_inventory_with_depth).
  Step 3: Qwen2.5-VL judges per class (JSON schema with per-item
          keep + reason) which classes the panorama seam should
          route around (stitcher.vlm_judge.judge_inventory).

The discovery script is decoupled from the main pipeline so that:
  - the pipeline's deps stay minimal (no ollama / transformers
    needed unless the user explicitly invokes discovery)
  - the JSON file is reusable across pipeline runs
  - the model env (ollama daemon, weights, transformers cache) can
    live somewhere else entirely

Entry points:
  * tools/suggest_fg_classes_v2.py  -- runs the discovery, writes
    auto_fg_classes.json (or a custom path), and also prints the
    kept-class list to stdout for piping.
  * video_stitcher_seam_gpu.py with --yoloe_fg_classes auto --
    reads the JSON via load_classes_from_json() below; falls back
    to a hardcoded list if the file is missing or malformed.
"""

import json
import os

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

# Default path for the JSON written by tools/suggest_fg_classes_v2.py
# and read by the main pipeline. Relative to the user's cwd.
DEFAULT_AUTO_FG_JSON = "auto_fg_classes.json"


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
            force a known vocabulary.
        return_details: if True, also return
            (vocab, records, decisions, raw_vlm_text) where
            decisions is the full per-item judge output (kept +
            dropped + reasons) and raw_vlm_text is the judge's
            JSON string response.

    Returns:
        list of class names to feed --yoloe_fg_classes (e.g.
        ['blue armchair', 'wooden desk', 'dual monitor setup', ...]).
        If return_details is True:
            (classes, vocab, records, decisions, raw_vlm_text).

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
    classes, decisions, raw = judge_inventory(
        frame_bgr, records,
        model_name=ollama_model,
        return_details=True,
    )

    if return_details:
        return classes, vocab, records, decisions, raw
    return classes


# ---------------------------------------------------------------------------
# JSON I/O for the pipeline <-> discovery hand-off
# ---------------------------------------------------------------------------


def load_classes_from_json(path, fallback_classes):
    """
    Load `kept_classes` from a JSON file produced by
    tools/suggest_fg_classes_v2.py.

    Returns `fallback_classes` if:
      - the file doesn't exist
      - the file isn't valid JSON
      - the top-level "kept_classes" key is missing
      - "kept_classes" isn't a non-empty list of strings

    Prints a one-line diagnostic to stderr for each failure mode so
    the user can tell *why* the fallback kicked in.
    """
    import sys

    if not os.path.isfile(path):
        print(f"[auto-fg-v2] no JSON at {path!r}; using hardcoded "
              f"fallback {fallback_classes}", file=sys.stderr)
        return list(fallback_classes)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[auto-fg-v2] could not parse {path!r} ({e}); using "
              f"hardcoded fallback {fallback_classes}", file=sys.stderr)
        return list(fallback_classes)

    kept = data.get("kept_classes")
    if (not isinstance(kept, list)
            or not kept
            or not all(isinstance(c, str) for c in kept)):
        print(f"[auto-fg-v2] {path!r} has no usable kept_classes; "
              f"using hardcoded fallback {fallback_classes}",
              file=sys.stderr)
        return list(fallback_classes)

    print(f"[auto-fg-v2] loaded {len(kept)} class(es) from {path!r}: "
          f"{kept}", file=sys.stderr)
    return list(kept)
