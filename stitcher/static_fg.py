"""
Depth-aware static-FG class selector.

Builds the final YOLOE text-class list for the main stitching pipeline
from two hand-curated lists:

  ALWAYS_KEEP        -- classes that go into the vocab unconditionally
                        (TV, monitor, picture frame, ...). Things a
                        seam should never cut through regardless of
                        where they sit in the scene.

  FOREGROUND_ONLY    -- classes that go in only if at least one
                        instance is in the foreground of frame 0
                        (chair, desk, plant, ...). When such an
                        object is on the back wall, a seam through
                        it is invisible; when it's right in front of
                        the camera, parallax breaks it.

The "non-important" tier is implicit: any class not in either list
is simply not asked of YOLOE.

The selection is a STARTUP-ONLY decision: depth is computed once on
frame 0, the final vocab is fixed for the run. Trade-off: tier-2
classes that aren't visible in frame 0 won't be tracked later in the
video. Acceptable for a static-camera setup where the room layout is
established up front; if a class needs to be tracked unconditionally,
move it to ALWAYS_KEEP.
"""

import gc

from stitcher.depth_estimation import (
    bbox_median_depth,
    estimate_depth,
    normalize_depth,
    release_depth,
)


# ---------------------------------------------------------------------------
# Editable tier lists
# ---------------------------------------------------------------------------
#
# These lists are the only knobs. Edit them directly to add / remove
# classes; they're picked up via stitcher.static_fg.* imports.
#
# Class names are YOLOE text prompts (any reasonable noun phrase).
# Multi-word phrases are fine (YOLOE matches via Mobile-CLIP).

ALWAYS_KEEP = [
    "tv",
    "monitor",
    "computer",
    "laptop",
    "screen",
    "picture frame",
    "poster",
    "whiteboard",
]

FOREGROUND_ONLY = [
    "chair",
    "armchair",
    "office chair",
    "stool",
    "couch",
    "sofa",
    "desk",
    "table",
    "coffee table",
    "dining table",
    "side table",
    "bookshelf",
    "cabinet",
    "plant",
    "potted plant",
    "houseplant",
    "lamp",
    "floor lamp",
]

# Normalized-depth threshold for the FOREGROUND_ONLY tier. Depth is
# in [0, 1] where 1.0 = closest to the camera, 0.0 = farthest (see
# stitcher.depth_estimation.normalize_depth). 0.4 lets through
# clearly-foreground instances and drops mid/background ones.
FG_DEPTH_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select_fg_classes_static(frame_bgr,
                             always_keep=None,
                             foreground_only=None,
                             depth_threshold=None,
                             yoloe_weights="yoloe-11s-seg.pt",
                             device="cuda:0",
                             min_confidence=0.0,
                             return_details=False):
    """
    Build the final YOLOE text-class vocabulary for the main pipeline.

    Algorithm:
      1. Run YOLOE on `frame_bgr` with vocab = always_keep + foreground_only.
      2. Run Depth Anything V2 once on the same frame; normalize to [0, 1].
      3. Final vocab = always_keep (unconditional) + every class in
         foreground_only with at least one detection whose bbox-median
         depth > depth_threshold.

    Args:
        frame_bgr: HxWx3 uint8 BGR image. Typically frame 0 of the
            left camera.
        always_keep: list of YOLOE text prompts. Defaults to ALWAYS_KEEP.
        foreground_only: list of YOLOE text prompts. Defaults to
            FOREGROUND_ONLY.
        depth_threshold: float in [0, 1]; classes in foreground_only
            need at least one instance with depth above this to be
            kept. Defaults to FG_DEPTH_THRESHOLD.
        yoloe_weights: YOLOE weights path.
        device: torch device for YOLOE + depth.
        min_confidence: drop YOLOE detections below this score before
            the depth check.
        return_details: if True, also return a list of per-class
            verdicts:
              [{"class": str,
                "tier": "always" | "foreground_only",
                "kept": bool,
                "max_depth": float | None,
                "reason": str},
               ...]
            Useful for the validator and audit logs.

    Returns:
        list of YOLOE text-class names to feed --yoloe_fg_classes.
        If return_details is True: (classes, verdicts).

    Frees both YOLOE and the depth pipeline before returning so the
    main pipeline can boot without VRAM contention.
    """
    if always_keep is None:
        always_keep = ALWAYS_KEEP
    if foreground_only is None:
        foreground_only = FOREGROUND_ONLY
    if depth_threshold is None:
        depth_threshold = FG_DEPTH_THRESHOLD

    # Combined vocab for the single YOLOE call. De-dup defensively;
    # if a class appears in both lists, treat it as always-keep.
    fg_only_dedup = [c for c in foreground_only if c not in always_keep]
    combined_vocab = list(always_keep) + fg_only_dedup

    # --- Step 1: YOLOE on the combined vocab.
    boxes_by_cid = _detect_with_yoloe(
        frame_bgr,
        text_classes=combined_vocab,
        yoloe_weights=yoloe_weights,
        device=device,
        min_confidence=min_confidence,
    )

    # --- Step 2: depth (only needed when at least one tier-2 class
    # was actually detected; otherwise we can skip loading the model).
    fg_only_cids = {
        i for i, name in enumerate(combined_vocab)
        if name in fg_only_dedup
    }
    fg_only_detected = any(boxes_by_cid.get(cid)
                           for cid in fg_only_cids)

    depth_norm = None
    if fg_only_detected:
        depth_raw = estimate_depth(frame_bgr)
        depth_norm = normalize_depth(depth_raw)
    release_depth()

    # --- Step 3: build per-class verdicts.
    verdicts = []
    kept = []

    for name in always_keep:
        verdicts.append({
            "class": name,
            "tier": "always",
            "kept": True,
            "max_depth": None,
            "reason": "always-keep tier",
        })
        kept.append(name)

    for name in fg_only_dedup:
        cid = combined_vocab.index(name)
        boxes = boxes_by_cid.get(cid, [])
        if not boxes:
            verdicts.append({
                "class": name,
                "tier": "foreground_only",
                "kept": False,
                "max_depth": None,
                "reason": "not detected in frame 0",
            })
            continue
        # Take the max depth across this class's instances. If any
        # one instance is foreground, the class is in the vocab.
        per_box = [
            bbox_median_depth(depth_norm, (b[0], b[1], b[2], b[3]))
            for b in boxes
        ]
        per_box = [d for d in per_box
                   if d == d]  # drop NaNs (degenerate bboxes)
        if not per_box:
            verdicts.append({
                "class": name,
                "tier": "foreground_only",
                "kept": False,
                "max_depth": None,
                "reason": "depth lookup failed for every bbox",
            })
            continue
        max_d = max(per_box)
        if max_d > depth_threshold:
            verdicts.append({
                "class": name,
                "tier": "foreground_only",
                "kept": True,
                "max_depth": max_d,
                "reason": f"foreground (d={max_d:.2f} > "
                          f"{depth_threshold:.2f})",
            })
            kept.append(name)
        else:
            verdicts.append({
                "class": name,
                "tier": "foreground_only",
                "kept": False,
                "max_depth": max_d,
                "reason": f"background (d={max_d:.2f} <= "
                          f"{depth_threshold:.2f})",
            })

    if return_details:
        return kept, verdicts
    return kept


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_with_yoloe(frame_bgr, text_classes,
                       yoloe_weights, device, min_confidence):
    """
    Run YOLOE in open-vocabulary mode on `text_classes`, return
    {class_id: [(x1, y1, x2, y2, conf), ...]}. The class_id ordering
    matches `text_classes`. Releases YOLOE's GPU memory before
    returning.
    """
    from stitcher.segmentation import PersonSegmenter

    seg = PersonSegmenter(
        yoloe_weights, device=device,
        use_yoloe=True, text_classes=list(text_classes),
    )
    try:
        class_ids = list(range(len(text_classes)))
        boxes_by_cid = seg.predict_classes_boxes(
            frame_bgr, class_ids=class_ids,
        )
    finally:
        del seg
        _free_gpu_memory()

    if min_confidence <= 0.0:
        return boxes_by_cid
    out = {}
    for cid, boxes in boxes_by_cid.items():
        kept = [b for b in boxes if b[4] >= min_confidence]
        if kept:
            out[cid] = kept
    return out


def _free_gpu_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
