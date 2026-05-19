"""
Standalone validator for PersonSegmenter.predict_classes_boxes.

Reads frame 0 of a video, runs YOLOE detection with a small text-
class list, and prints the per-class bounding boxes. Also draws
them on the frame and saves the annotated image so you can visually
confirm the boxes line up with the actual objects.

Usage:
    python tools/test_boxes.py \\
        --video videos/sf_left.mp4 \\
        --classes "yellow chair" "blue rug" "table" "plant"

Output:
    videos/sf_left_boxes.png  (annotated frame with bboxes drawn)
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2  # noqa: E402

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


def _draw_boxes(frame_bgr, boxes_by_class, class_names):
    """Annotate frame in place with each class's bboxes + labels."""
    out = frame_bgr.copy()
    # Cycle through a few distinct colours.
    palette = [(0, 255, 255), (0, 255, 0), (255, 0, 255),
               (255, 255, 0), (0, 128, 255), (255, 128, 0)]
    for cid, boxes in boxes_by_class.items():
        colour = palette[cid % len(palette)]
        name = class_names[cid] if cid < len(class_names) else f"class_{cid}"
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
        description="Run YOLOE on frame 0 of a video with a custom class "
                    "list; print per-class bboxes and save an annotated "
                    "image for visual inspection."
    )
    parser.add_argument("--video", required=True,
                        help="Path to a video; frame 0 is read.")
    parser.add_argument("--classes", required=True, nargs="+",
                        help="YOLOE text classes (quote multi-word ones).")
    parser.add_argument("--yoloe_weights", default="yoloe-11s-seg.pt",
                        help="YOLOE weights path. Default: yoloe-11s-seg.pt.")
    parser.add_argument("--device", default="cuda:0",
                        help="Torch device. Default: cuda:0.")
    parser.add_argument("--output", default=None,
                        help="Output PNG path. Default: <video>_boxes.png.")
    args = parser.parse_args()

    print(f"[test_boxes] reading frame 0 from {args.video}")
    frame = _grab_frame_zero(args.video)
    H, W = frame.shape[:2]
    print(f"[test_boxes] frame shape: H={H} W={W}")
    print(f"[test_boxes] classes: {args.classes}")

    print(f"[test_boxes] loading YOLOE from {args.yoloe_weights}...")
    seg = PersonSegmenter(
        args.yoloe_weights, device=args.device,
        use_yoloe=True, text_classes=args.classes,
    )

    class_ids = list(range(len(args.classes)))
    boxes_by_class = seg.predict_classes_boxes(frame, class_ids=class_ids)

    print(f"[test_boxes] detections:")
    total = 0
    for cid in class_ids:
        name = args.classes[cid]
        boxes = boxes_by_class.get(cid, [])
        total += len(boxes)
        if boxes:
            print(f"  {name!r}: {len(boxes)} detection(s)")
            for (x1, y1, x2, y2, conf) in boxes:
                print(f"    bbox=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}) "
                      f"conf={conf:.3f}")
        else:
            print(f"  {name!r}: (no detections)")
    print(f"[test_boxes] total: {total} detection(s) across "
          f"{len(args.classes)} class(es)")

    annotated = _draw_boxes(frame, boxes_by_class, args.classes)
    out_path = args.output or (
        os.path.splitext(args.video)[0] + "_boxes.png"
    )
    cv2.imwrite(out_path, annotated)
    print(f"[test_boxes] saved annotated frame -> {out_path}")


if __name__ == "__main__":
    main()
