"""
Standalone validator for stitcher.object_inventory.detect_all_objects.

Reads frame 0 of a video, runs YOLOE with the comprehensive indoor-
object vocabulary, prints what it found, and saves an annotated
image with every detected bbox drawn so you can eyeball coverage
on your scene.

Usage:
    python tools/test_inventory.py --video videos/sf_left.mp4

If the default vocabulary misses something obvious in your room,
add it directly to INVENTORY_CLASSES in stitcher/object_inventory.py
(or pass --extra "foo" "bar" to extend without editing).
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2  # noqa: E402

from stitcher.object_inventory import (  # noqa: E402
    INVENTORY_CLASSES,
    detect_all_objects,
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


def _draw_inventory(frame_bgr, detections):
    """Annotate frame with all detected bboxes + class labels."""
    out = frame_bgr.copy()
    # Cycle through high-contrast colours.
    palette = [
        (0, 255, 255), (0, 255, 0), (255, 0, 255), (255, 255, 0),
        (0, 128, 255), (255, 128, 0), (128, 0, 255), (0, 255, 128),
        (255, 0, 128), (128, 255, 0),
    ]
    for i, (name, boxes) in enumerate(sorted(detections.items())):
        colour = palette[i % len(palette)]
        for (x1, y1, x2, y2, conf) in boxes:
            p1 = (int(round(x1)), int(round(y1)))
            p2 = (int(round(x2)), int(round(y2)))
            cv2.rectangle(out, p1, p2, colour, 2)
            label = f"{name} {conf:.2f}"
            cv2.putText(out, label, (p1[0], max(0, p1[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Run object_inventory.detect_all_objects on frame 0 "
                    "of a video; print what was found and save an "
                    "annotated image with all bboxes."
    )
    parser.add_argument("--video", required=True,
                        help="Path to a video; frame 0 is read.")
    parser.add_argument("--yoloe_weights", default="yoloe-11s-seg.pt",
                        help="YOLOE weights path.")
    parser.add_argument("--device", default="cuda:0",
                        help="Torch device. Default: cuda:0.")
    parser.add_argument("--min_confidence", type=float, default=0.0,
                        help="Drop detections below this score. "
                             "Default 0.0 (keep all).")
    parser.add_argument("--extra", nargs="+", default=None,
                        help="Extra text classes to append to the "
                             "default INVENTORY_CLASSES vocabulary.")
    parser.add_argument("--output", default=None,
                        help="Output PNG path. Default: <video>_inventory.png.")
    args = parser.parse_args()

    classes = list(INVENTORY_CLASSES)
    if args.extra:
        classes.extend(args.extra)

    print(f"[test_inventory] reading frame 0 from {args.video}")
    frame = _grab_frame_zero(args.video)
    H, W = frame.shape[:2]
    print(f"[test_inventory] frame shape: H={H} W={W}")
    print(f"[test_inventory] running YOLOE over {len(classes)} classes...")

    detections = detect_all_objects(
        frame,
        yoloe_weights=args.yoloe_weights,
        device=args.device,
        class_list=classes,
        min_confidence=args.min_confidence,
    )

    total = sum(len(v) for v in detections.values())
    print(f"[test_inventory] detected {total} object(s) across "
          f"{len(detections)} class(es):")
    for name in sorted(detections):
        boxes = detections[name]
        confs = [f"{b[4]:.2f}" for b in boxes]
        print(f"  {name!r}: {len(boxes)} (conf: {', '.join(confs)})")

    # Also flag classes from the vocab that found nothing (briefly,
    # for visibility).
    found_set = set(detections)
    not_found = [c for c in classes if c not in found_set]
    print(f"[test_inventory] not detected: {len(not_found)} class(es) "
          f"({', '.join(not_found[:8])}"
          f"{', ...' if len(not_found) > 8 else ''})")

    annotated = _draw_inventory(frame, detections)
    out_path = args.output or (
        os.path.splitext(args.video)[0] + "_inventory.png"
    )
    cv2.imwrite(out_path, annotated)
    print(f"[test_inventory] saved annotated frame -> {out_path}")


if __name__ == "__main__":
    main()
