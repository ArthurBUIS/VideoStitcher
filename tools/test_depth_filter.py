"""
Standalone validator for the depth-aware FG/BG filter.

End-to-end on one frame:
    1. Estimate depth (Depth Anything V2 Small).
    2. Detect bboxes for the user's class list (YOLOE).
    3. Classify each class as 'fg' or 'bg' by bbox-median depth
       relative to the scene median (or another percentile).
    4. Print results.
    5. Save an annotated image: FG bboxes in green, BG bboxes in red.

Use this BEFORE wiring the filter into suggest_fg_classes -- you
can iterate on --threshold and see immediately which classes end
up FG vs BG.

Usage:
    python tools/test_depth_filter.py \\
        --video videos/sf_left.mp4 \\
        --classes "yellow chair" "blue rug" "table" "plant" \\
        --threshold 1.0
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from stitcher.depth_fg import (  # noqa: E402
    classify_classes_by_depth,
    estimate_depth,
)
from stitcher.segmentation import PersonSegmenter  # noqa: E402


def _grab_frame_zero(path):
    cap = cv2.VideoCapture(path)
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame 0 from {path}")
    return frame


def _draw_classified_boxes(frame_bgr, boxes_by_class, labels, class_names):
    """Annotate frame: FG bboxes green, BG bboxes red, undetected omitted.
    labels: {class_id: 'fg' | 'bg'} from classify_classes_by_depth."""
    out = frame_bgr.copy()
    for cid, boxes in boxes_by_class.items():
        if not boxes:
            continue
        label = labels.get(cid)
        if label is None:
            continue
        colour = (0, 200, 0) if label == "fg" else (0, 0, 200)
        name = class_names[cid] if cid < len(class_names) else f"class_{cid}"
        for (x1, y1, x2, y2, conf) in boxes:
            p1 = (int(round(x1)), int(round(y1)))
            p2 = (int(round(x2)), int(round(y2)))
            cv2.rectangle(out, p1, p2, colour, 2)
            tag = f"{name} [{label.upper()}] {conf:.2f}"
            cv2.putText(out, tag, (p1[0], max(0, p1[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end test of depth-aware FG filtering on one "
                    "video frame. Saves an annotated PNG showing each "
                    "class's bboxes coloured by FG/BG decision."
    )
    parser.add_argument("--video", required=True,
                        help="Path to a video; frame 0 is read.")
    parser.add_argument("--classes", required=True, nargs="+",
                        help="YOLOE text classes to test (quote multi-word).")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="bbox is 'fg' if bbox-median depth > "
                             "scene_reference * threshold. Default 1.0 "
                             "(closer than scene median). Higher "
                             "(1.1, 1.2) = stricter; lower (0.9) = looser.")
    parser.add_argument("--reference_percentile", type=float, default=50.0,
                        help="Percentile of the depth map used as the "
                             "scene reference. Default 50 (median).")
    parser.add_argument("--yoloe_weights", default="yoloe-11s-seg.pt",
                        help="YOLOE weights path.")
    parser.add_argument("--device", default="cuda:0",
                        help="Torch device. Default: cuda:0.")
    parser.add_argument("--output", default=None,
                        help="Output PNG path. Default: <video>_filter.png.")
    args = parser.parse_args()

    print(f"[test_depth_filter] reading frame 0 from {args.video}")
    frame = _grab_frame_zero(args.video)
    H, W = frame.shape[:2]
    print(f"[test_depth_filter] frame shape: H={H} W={W}")
    print(f"[test_depth_filter] classes: {args.classes}")
    print(f"[test_depth_filter] threshold={args.threshold} "
          f"reference_percentile={args.reference_percentile}")

    print("[test_depth_filter] estimating depth (first run loads model)...")
    depth = estimate_depth(frame)
    scene_ref = float(np.percentile(depth, args.reference_percentile))
    print(f"[test_depth_filter] depth stats: min={depth.min():.3f} "
          f"max={depth.max():.3f} ref={scene_ref:.3f}")

    print(f"[test_depth_filter] loading YOLOE from {args.yoloe_weights}...")
    seg = PersonSegmenter(
        args.yoloe_weights, device=args.device,
        use_yoloe=True, text_classes=args.classes,
    )
    class_ids = list(range(len(args.classes)))
    boxes_by_class = seg.predict_classes_boxes(frame, class_ids=class_ids)

    labels = classify_classes_by_depth(
        boxes_by_class, depth,
        threshold=args.threshold,
        reference_percentile=args.reference_percentile,
    )

    print("[test_depth_filter] per-class decisions:")
    for cid in class_ids:
        name = args.classes[cid]
        boxes = boxes_by_class.get(cid, [])
        if not boxes:
            print(f"  {name!r}: (no detections, omitted)")
            continue
        label = labels.get(cid, "?")
        # Show bbox depth medians for transparency.
        per_box_medians = []
        for (x1, y1, x2, y2, _conf) in boxes:
            xi1 = max(0, min(W, int(round(x1))))
            xi2 = max(0, min(W, int(round(x2))))
            yi1 = max(0, min(H, int(round(y1))))
            yi2 = max(0, min(H, int(round(y2))))
            region = depth[yi1:yi2, xi1:xi2]
            if region.size:
                per_box_medians.append(float(np.median(region)))
        medians_str = ", ".join(f"{m:.3f}" for m in per_box_medians)
        print(f"  {name!r}: {label.upper()}  "
              f"(bbox medians: [{medians_str}]  ref={scene_ref:.3f})")

    annotated = _draw_classified_boxes(
        frame, boxes_by_class, labels, args.classes,
    )
    out_path = args.output or (
        os.path.splitext(args.video)[0] + "_filter.png"
    )
    cv2.imwrite(out_path, annotated)
    print(f"[test_depth_filter] saved annotated frame -> {out_path}")
    print("[test_depth_filter] visual check: GREEN = foreground, "
          "RED = background.")


if __name__ == "__main__":
    main()
