"""
Standalone validator for stitcher.vlm_judge.judge_inventory.

Runs Step 1 (object inventory) + Step 2 (depth annotation) + Step 3
(VLM judge) on frame 0 of a video. Prints:
  - the depth-annotated inventory the VLM was shown
  - the VLM's raw response
  - the parsed kept-classes list

and saves two PNGs:
  - <video>_vlm_judge_all.png     -- every detected bbox, dimmed
  - <video>_vlm_judge_kept.png    -- only bboxes whose class the VLM
                                     chose, drawn in colour

Usage:
    python tools/test_vlm_judge.py --video videos/sf_left.mp4

Iterate on stitcher/vlm_judge.py's JUDGE_PROMPT_TEMPLATE until the
kept list matches what you would have picked by eye, then move on.
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
from stitcher.vlm_judge import (  # noqa: E402
    DEFAULT_OLLAMA_MODEL,
    judge_inventory,
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


def _draw_records(frame_bgr, records, kept_set=None, dim_others=False):
    """
    Draw bboxes + class labels.
      - If kept_set is None: draw everything in a per-class colour.
      - If kept_set is set: draw kept classes in bright colour, draw
        the rest faint grey (or skip them entirely if dim_others=False).
    """
    out = frame_bgr.copy()
    palette = [
        (0, 255, 255), (0, 255, 0), (255, 0, 255), (255, 255, 0),
        (0, 128, 255), (255, 128, 0), (128, 0, 255), (0, 255, 128),
        (255, 0, 128), (128, 255, 0),
    ]
    # Stable colour per class name.
    seen = {}

    def colour_for(name):
        if name not in seen:
            seen[name] = palette[len(seen) % len(palette)]
        return seen[name]

    for r in records:
        name = r["class"]
        x1, y1, x2, y2 = (int(round(v)) for v in r["bbox"])
        if kept_set is not None and name not in kept_set:
            if not dim_others:
                continue
            colour = (90, 90, 90)
            thickness = 1
        else:
            colour = colour_for(name)
            thickness = 2
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness)
        label = f"{name} d={r['depth']:.2f}"
        cv2.putText(out, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, thickness)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Run object_inventory + depth + VLM judge on "
                    "frame 0; print the kept classes and save two "
                    "annotated PNGs (all bboxes vs. only kept-class "
                    "bboxes)."
    )
    parser.add_argument("--video", required=True,
                        help="Path to a video; frame 0 is read.")
    parser.add_argument("--yoloe_weights", default="yoloe-11s-seg.pt",
                        help="YOLOE weights path.")
    parser.add_argument("--device", default="cuda:0",
                        help="Torch device. Default: cuda:0.")
    parser.add_argument("--ollama_model", default=DEFAULT_OLLAMA_MODEL,
                        help="Ollama model tag for the VLM judge.")
    parser.add_argument("--min_confidence", type=float, default=0.0,
                        help="Drop YOLOE detections below this score.")
    parser.add_argument("--output_prefix", default=None,
                        help="Output PNG prefix. Default: <video>_vlm_judge.")
    args = parser.parse_args()

    print(f"[test_vlm_judge] reading frame 0 from {args.video}")
    frame = _grab_frame_zero(args.video)
    H, W = frame.shape[:2]
    print(f"[test_vlm_judge] frame shape: H={H} W={W}")

    print("[test_vlm_judge] step 1: YOLOE object inventory...")
    detections = detect_all_objects(
        frame,
        yoloe_weights=args.yoloe_weights,
        device=args.device,
        min_confidence=args.min_confidence,
    )
    n_det = sum(len(v) for v in detections.values())
    print(f"[test_vlm_judge]   {n_det} detection(s) across "
          f"{len(detections)} class(es)")

    print("[test_vlm_judge] step 2: depth annotation...")
    records = annotate_inventory_with_depth(frame, detections)
    print(f"[test_vlm_judge]   {len(records)} record(s) (closest first):")
    for r in records:
        print(f"    d={r['depth']:.2f}  conf={r['conf']:.2f}  "
              f"{r['class']}")

    print(f"[test_vlm_judge] step 3: VLM judge ({args.ollama_model})...")
    kept, decisions, raw = judge_inventory(
        frame, records,
        model_name=args.ollama_model,
        return_details=True,
    )
    print(f"[test_vlm_judge]   per-class decisions ({len(decisions)}):")
    for d in decisions:
        flag = "KEEP" if d["keep"] else "drop"
        print(f"    [{flag}] {d['name']:<22s} {d['reason']}")
    print(f"[test_vlm_judge]   kept classes ({len(kept)}): {kept}")
    print("[test_vlm_judge]   raw VLM response (JSON):")
    print("    " + raw.replace("\n", "\n    "))

    prefix = args.output_prefix or (
        os.path.splitext(args.video)[0] + "_vlm_judge"
    )
    all_png = prefix + "_all.png"
    kept_png = prefix + "_kept.png"

    cv2.imwrite(all_png, _draw_records(frame, records))
    print(f"[test_vlm_judge] saved (all bboxes)   -> {all_png}")

    kept_set = set(kept)
    cv2.imwrite(kept_png,
                _draw_records(frame, records,
                              kept_set=kept_set, dim_others=True))
    print(f"[test_vlm_judge] saved (kept bboxes)  -> {kept_png}")


if __name__ == "__main__":
    main()
