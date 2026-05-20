"""
Detect every recognisable object in a single camera frame, using
YOLOE as the open-vocabulary detector with a comprehensive indoor-
object vocabulary.

This is Step A of the auto-FG-v2 discovery flow: produce a complete
inventory of what's in the room (foreground AND background, free-
standing AND wall-mounted), so the downstream depth + VLM judgment
steps can pick which classes the panorama seam should respect.

The vocabulary lives in INVENTORY_CLASSES below. It's a public
module-level constant -- edit it directly, or pass `class_list=...`
to override per-call. A future enhancement might replace this
hardcoded list with a VLM-generated one per scene (see the
discussion in the auto_fg_v2 design notes).
"""

import gc


# Comprehensive indoor-object vocabulary. ~40 entries; kept short
# enough to fit comfortably in YOLOE's text-embedding budget on a
# T1000 without slowing inference noticeably. Edit / extend
# whenever a scene has objects that aren't here.
INVENTORY_CLASSES = [
    # Seating
    "chair", "armchair", "office chair", "stool",
    "couch", "sofa", "bench", "ottoman",
    # Tables and surfaces
    "table", "coffee table", "desk", "side table", "dining table",
    # Storage furniture
    "bookshelf", "shelf", "cabinet", "dresser", "wardrobe",
    # Electronics
    "monitor", "tv", "laptop", "keyboard", "speaker",
    # Lighting
    "lamp", "floor lamp", "desk lamp",
    # Wall / surface decor
    "picture frame", "painting", "mirror", "poster", "clock", "vase",
    # Plants
    "plant", "potted plant", "houseplant",
    # Floor coverings
    "rug", "carpet", "mat",
    # Misc indoor objects
    "box", "basket", "trash can", "whiteboard",
]


def _free_gpu_memory():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def detect_all_objects(frame_bgr,
                       yoloe_weights="yoloe-11s-seg.pt",
                       device="cuda:0",
                       class_list=None,
                       min_confidence=0.0):
    """
    Run YOLOE on the given BGR frame with a comprehensive indoor-
    object text vocabulary; return per-class bounding boxes.

    Args:
        frame_bgr: OpenCV-style HxWx3 uint8 BGR image.
        yoloe_weights: path to the YOLOE weights (default
            yoloe-11s-seg.pt -- auto-downloads on first use).
        device: torch device. Default "cuda:0".
        class_list: text classes YOLOE should look for. Defaults to
            INVENTORY_CLASSES; pass a custom list to extend or
            replace.
        min_confidence: drop detections below this score. Default 0
            (keep everything).

    Returns: dict {class_name (str): [(x1, y1, x2, y2, conf), ...]}.
        Pixel coords in the input frame. Only classes with at least
        one surviving detection appear in the dict.

    YOLOE is freed after the call so its GPU memory doesn't pile up
    before any downstream model (e.g. Depth Anything V2 or the
    stitching pipeline's own YOLOE) loads.
    """
    if class_list is None:
        class_list = INVENTORY_CLASSES

    # Lazy import so callers that don't need YOLOE don't pay for the
    # ultralytics dependency.
    from stitcher.segmentation import PersonSegmenter

    seg = PersonSegmenter(
        yoloe_weights, device=device,
        use_yoloe=True, text_classes=list(class_list),
    )
    try:
        class_ids = list(range(len(class_list)))
        boxes_by_cid = seg.predict_classes_boxes(
            frame_bgr, class_ids=class_ids,
        )
    finally:
        del seg
        _free_gpu_memory()

    out = {}
    for cid, boxes in boxes_by_cid.items():
        if not boxes:
            continue
        kept = [b for b in boxes if b[4] >= min_confidence]
        if kept:
            out[class_list[cid]] = kept
    return out
