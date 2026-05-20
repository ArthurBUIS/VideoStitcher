"""
Standalone validator for stitcher.depth_inventory.annotate_inventory_with_depth.

Combines Step 1 (object inventory) + Step 2 (per-bbox depth). Reads
frame 0 of a video, detects every object, annotates each with a
depth value, prints a list sorted closest -> farthest, and saves an
annotated image where each bbox is colour-coded by depth (red =
close, blue = far) with a "class d=0.xx" label.

Usage:
    python tools/test_depth_inventory.py --video videos/sf_left.mp4

If a bbox has a depth value that doesn't match what you see in the
scene, the depth model is the suspect -- not the bbox.
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2  # noqa: E402

from stitcher.object_inventory import detect_all_objects  # noqa: E402
from stitcher.depth_inventory import (  # noqa: E402
    annotate_inventory_with_depth,
)


def _grab_frame_zero(path):
    cap = cv2.VideoCapture(path)
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame 0 from {path}")
    return frame


def _depth_to_bgr(d):
    """Red (closest, d=1) -> Blue (farthest, d=0)."""
    b = int(round((1.0 - d) * 255))
    r = int(round(d * 255))
    return (b, 0, r)


def _draw_annotated(frame_bgr, records):
    """Draw bboxes coloured by depth, with class + depth labels."""
    out = frame_bgr.copy()
    H = out.shape[0]
    for r in records:
        colour = _depth_to_bgr(r["depth"])
        x1, y1, x2, y2 = (int(round(v)) for v in r["bbox"])
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        label = f"{r['class']} d={r['depth']:.2f}"
        y_text = max(15, y1 - 6)
        cv2.putText(out, label, (x1, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Run object_inventory + depth annotation on "
                    "frame 0 of a video; print a depth-sorted list "
                    "and save an annotated image."
    )
    parser.add_argument("--video", required=True,
                        help="Path to a video; frame 0 is read.")
    parser.add_argument("--yoloe_weights", default="yoloe-11s-seg.pt",
                        help="YOLOE weights path.")
    parser.add_argument("--device", default="cuda:0",
                        help="Torch device. Default: cuda:0.")
    parser.add_argument("--min_confidence", type=float, default=0.0,
                        help="Drop YOLOE detections below this score.")
    parser.add_argument("--output", default=None,
                        help="Output PNG. Default: "
                             "<video>_depth_inventory.png.")
    args = parser.parse_args()

    print(f"[test_depth_inventory] reading frame 0 from {args.video}")
    frame = _grab_frame_zero(args.video)
    H, W = frame.shape[:2]
    print(f"[test_depth_inventory] frame shape: H={H} W={W}")

    print("[test_depth_inventory] running YOLOE object inventory...")
    detections = detect_all_objects(
        frame,
        yoloe_weights=args.yoloe_weights,
        device=args.device,
        min_confidence=args.min_confidence,
    )
    n_det = sum(len(v) for v in detections.values())
    print(f"[test_depth_inventory] {n_det} detection(s) across "
          f"{len(detections)} class(es)")

    print("[test_depth_inventory] estimating depth + annotating...")
    records = annotate_inventory_with_depth(frame, detections)

    print(f"[test_depth_inventory] {len(records)} record(s), "
          f"sorted by depth (closest first):")
    for r in records:
        x1, y1, x2, y2 = (int(round(v)) for v in r["bbox"])
        print(f"  d={r['depth']:.2f}  conf={r['conf']:.2f}  "
              f"{r['class']!r:<20s} bbox=({x1},{y1},{x2},{y2})")

    annotated = _draw_annotated(frame, records)
    out_path = args.output or (
        os.path.splitext(args.video)[0] + "_depth_inventory.png"
    )
    cv2.imwrite(out_path, annotated)
    print(f"[test_depth_inventory] saved annotated frame -> {out_path}")


if __name__ == "__main__":
    main()
