"""
Standalone validator for the runtime depth-filter behavior.

Simulates what the pipeline does at every FG recompute tick:
runs YOLOE on the combined ALWAYS_KEEP + FOREGROUND_ONLY vocab over
frame 0 of a video, runs depth on the same frame, then walks each
detection and prints whether it would be kept or dropped:

    [keep] tv               (always-keep)     d=n/a
    [keep] chair            (foreground)      d=0.82 > 0.40
    [drop] chair            (foreground)      d=0.21 <= 0.40
    [drop] plant            (foreground)      d=0.05 <= 0.40

Two siblings of the same class CAN now have different verdicts.
That's the whole point of the refactor.

Also saves an annotated PNG (`<video>_static_fg.png`) with kept-
detection bboxes in colour and dropped ones dimmed grey.

Usage:
    python tools/test_static_fg.py --video videos/sf_left.mp4
"""

import argparse
import gc
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
    get_combined_vocab,
    get_fg_only_indices,
)
from stitcher.depth_estimation import (  # noqa: E402
    bbox_median_depth,
    estimate_depth,
    normalize_depth,
    release_depth,
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


def _run_yoloe_boxes(frame_bgr, vocab, yoloe_weights, device,
                    min_confidence):
    """Run YOLOE once on `vocab` and return
    {class_id: [(x1, y1, x2, y2, conf), ...]}. Releases YOLOE GPU
    memory before returning."""
    from stitcher.segmentation import PersonSegmenter

    seg = PersonSegmenter(
        yoloe_weights, device=device,
        use_yoloe=True, text_classes=list(vocab),
    )
    try:
        boxes_by_cid = seg.predict_classes_boxes(
            frame_bgr, class_ids=list(range(len(vocab))),
        )
    finally:
        del seg
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    if min_confidence > 0:
        boxes_by_cid = {
            cid: [b for b in boxes if b[4] >= min_confidence]
            for cid, boxes in boxes_by_cid.items()
        }
    return boxes_by_cid


def _draw_annotated(frame_bgr, verdicts):
    """Draw every detection. Kept ones in per-class colour, dropped
    ones dim grey. `verdicts` is the per-detection list this script
    builds."""
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

    for v in verdicts:
        x1, y1, x2, y2 = (int(round(c)) for c in v["bbox"])
        if v["kept"]:
            colour = colour_for(v["class"])
            thickness = 2
        else:
            colour = (90, 90, 90)
            thickness = 1
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness)
        if v["depth"] is not None:
            label = f"{v['class']} d={v['depth']:.2f}"
        else:
            label = v["class"]
        cv2.putText(out, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, thickness)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Simulate the runtime depth-filter on frame 0 "
                    "of a video; print per-detection verdicts and "
                    "save an annotated image."
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

    vocab = get_combined_vocab()
    fg_only_idx = set(get_fg_only_indices(vocab))
    print(f"[test_static_fg] running YOLOE on {len(vocab)} classes...")
    boxes_by_cid = _run_yoloe_boxes(
        frame, vocab,
        yoloe_weights=args.yoloe_weights,
        device=args.device,
        min_confidence=args.min_confidence,
    )
    n_det = sum(len(v) for v in boxes_by_cid.values())
    print(f"[test_static_fg] {n_det} detection(s)")

    fg_only_detected = any(
        boxes_by_cid.get(cid) for cid in fg_only_idx
    )
    depth_norm = None
    if fg_only_detected:
        print("[test_static_fg] running Depth Anything V2...")
        depth_norm = normalize_depth(estimate_depth(frame))
    release_depth()

    # Build per-detection verdicts and the kept-class summary.
    verdicts = []
    for cid, boxes in boxes_by_cid.items():
        name = vocab[cid]
        is_fg_only = cid in fg_only_idx
        for (x1, y1, x2, y2, conf) in boxes:
            if not is_fg_only:
                verdicts.append({
                    "class": name,
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf,
                    "depth": None,
                    "tier": "always-keep",
                    "kept": True,
                    "reason": "always-keep tier",
                })
                continue
            d = bbox_median_depth(depth_norm, (x1, y1, x2, y2))
            kept = (d == d) and (d > args.depth_threshold)
            if kept:
                reason = (f"foreground (d={d:.2f} > "
                          f"{args.depth_threshold:.2f})")
            else:
                d_str = "nan" if d != d else f"{d:.2f}"
                reason = (f"background (d={d_str} <= "
                          f"{args.depth_threshold:.2f})")
            verdicts.append({
                "class": name,
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
                "depth": None if d != d else d,
                "tier": "foreground",
                "kept": kept,
                "reason": reason,
            })

    # Sort: kept first, then by class name. Easier to scan.
    verdicts.sort(key=lambda v: (not v["kept"], v["class"]))
    print("[test_static_fg] per-detection verdicts:")
    for v in verdicts:
        flag = "keep" if v["kept"] else "drop"
        print(f"  [{flag}] {v['class']:<20s} ({v['tier']:<10s}) "
              f"{v['reason']}")

    kept_classes = sorted({v["class"] for v in verdicts if v["kept"]})
    print(f"[test_static_fg] kept classes ({len(kept_classes)}): "
          f"{kept_classes}")

    annotated = _draw_annotated(frame, verdicts)
    out_path = args.output or (
        os.path.splitext(args.video)[0] + "_static_fg.png"
    )
    cv2.imwrite(out_path, annotated)
    print(f"[test_static_fg] saved annotated frame -> {out_path}")


if __name__ == "__main__":
    main()
