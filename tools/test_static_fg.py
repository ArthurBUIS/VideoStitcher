"""
Standalone validator for stitcher.static_fg.select_fg_classes_static.

Runs the depth-aware tiered selector on frame 0 of a video and
prints per-class verdicts:

    [keep] tv               always-keep tier
    [keep] chair            foreground (d=0.82 > 0.40)
    [drop] plant            background (d=0.15 <= 0.40)
    [drop] couch            not detected in frame 0

Also saves an annotated PNG (`<video>_static_fg.png`) with the kept-
class bboxes drawn in colour and the dropped ones dimmed.

Usage:
    python tools/test_static_fg.py --video videos/sf_left.mp4

Iterate on stitcher/static_fg.py's ALWAYS_KEEP / FOREGROUND_ONLY
lists and FG_DEPTH_THRESHOLD until the kept set matches what you'd
have picked by eye.
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2  # noqa: E402

from stitcher.static_fg import (  # noqa: E402
    ALWAYS_KEEP,
    FG_DEPTH_THRESHOLD,
    FOREGROUND_ONLY,
    _detect_with_yoloe,
    select_fg_classes_static,
)
from stitcher.depth_estimation import (  # noqa: E402
    bbox_median_depth,
    estimate_depth,
    normalize_depth,
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


def _draw_annotated(frame_bgr, combined_vocab, boxes_by_cid,
                    depth_norm, kept_set):
    """
    Draw every detected bbox with its class + depth label. Kept-
    class bboxes are coloured; dropped ones are dimmed grey.
    """
    out = frame_bgr.copy()
    palette = [
        (0, 255, 255), (0, 255, 0), (255, 0, 255), (255, 255, 0),
        (0, 128, 255), (255, 128, 0), (128, 0, 255), (0, 255, 128),
        (255, 0, 128), (128, 255, 0),
    ]
    colour_by_name = {}

    def colour_for(name):
        if name not in colour_by_name:
            colour_by_name[name] = palette[len(colour_by_name)
                                           % len(palette)]
        return colour_by_name[name]

    for cid, boxes in boxes_by_cid.items():
        name = combined_vocab[cid]
        is_kept = name in kept_set
        for (x1, y1, x2, y2, conf) in boxes:
            d = bbox_median_depth(depth_norm,
                                  (x1, y1, x2, y2)) if depth_norm is not None else float("nan")
            if is_kept:
                colour = colour_for(name)
                thickness = 2
            else:
                colour = (90, 90, 90)
                thickness = 1
            p1 = (int(round(x1)), int(round(y1)))
            p2 = (int(round(x2)), int(round(y2)))
            cv2.rectangle(out, p1, p2, colour, thickness)
            label = (f"{name} d={d:.2f}"
                     if d == d else f"{name}")
            cv2.putText(out, label,
                        (p1[0], max(15, p1[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour,
                        thickness)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Run the depth-aware static-FG selector on "
                    "frame 0 of a video; print per-class verdicts "
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
    parser.add_argument("--depth_threshold", type=float,
                        default=FG_DEPTH_THRESHOLD,
                        help=f"Depth threshold for the FOREGROUND_ONLY "
                             f"tier (default {FG_DEPTH_THRESHOLD}). "
                             f"Higher = stricter foreground criterion.")
    parser.add_argument("--output", default=None,
                        help="Output PNG path. Default: "
                             "<video>_static_fg.png.")
    args = parser.parse_args()

    print(f"[test_static_fg] reading frame 0 from {args.video}")
    frame = _grab_frame_zero(args.video)
    H, W = frame.shape[:2]
    print(f"[test_static_fg] frame shape: H={H} W={W}")

    print(f"[test_static_fg] ALWAYS_KEEP ({len(ALWAYS_KEEP)}): "
          f"{ALWAYS_KEEP}")
    print(f"[test_static_fg] FOREGROUND_ONLY ({len(FOREGROUND_ONLY)}): "
          f"{FOREGROUND_ONLY}")
    print(f"[test_static_fg] depth threshold: {args.depth_threshold}")

    # Run the orchestrated selector for the final list + verdicts.
    kept, verdicts = select_fg_classes_static(
        frame,
        depth_threshold=args.depth_threshold,
        yoloe_weights=args.yoloe_weights,
        device=args.device,
        min_confidence=args.min_confidence,
        return_details=True,
    )

    print("[test_static_fg] per-class verdicts:")
    for v in verdicts:
        flag = "keep" if v["kept"] else "drop"
        tier = v["tier"]
        print(f"  [{flag}] {v['class']:<20s} ({tier}): {v['reason']}")

    print(f"[test_static_fg] final vocab ({len(kept)}): {kept}")

    # Re-run YOLOE + depth for the annotated image (cheap on frame 0,
    # and the orchestrator releases both models so we'd have to
    # reload anyway). This keeps test_static_fg.py independent from
    # the orchestrator's internals.
    print("[test_static_fg] rendering annotated image...")
    fg_only_dedup = [c for c in FOREGROUND_ONLY if c not in ALWAYS_KEEP]
    combined_vocab = list(ALWAYS_KEEP) + fg_only_dedup
    boxes_by_cid = _detect_with_yoloe(
        frame,
        text_classes=combined_vocab,
        yoloe_weights=args.yoloe_weights,
        device=args.device,
        min_confidence=args.min_confidence,
    )
    depth_norm = None
    if any(boxes_by_cid.get(i) for i in range(len(ALWAYS_KEEP),
                                              len(combined_vocab))):
        depth_norm = normalize_depth(estimate_depth(frame))

    annotated = _draw_annotated(frame, combined_vocab, boxes_by_cid,
                                depth_norm, set(kept))
    out_path = args.output or (
        os.path.splitext(args.video)[0] + "_static_fg.png"
    )
    cv2.imwrite(out_path, annotated)
    print(f"[test_static_fg] saved annotated frame -> {out_path}")


if __name__ == "__main__":
    main()
