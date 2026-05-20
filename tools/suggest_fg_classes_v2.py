"""
Standalone CLI for auto-FG-v2 discovery.

Runs Steps 1+2+3 on frame 0 of a video and prints the resulting
class list to stdout (comma-separated, no trailing punctuation).
Suitable for piping into --yoloe_fg_classes, or just for confirming
what `--yoloe_fg_classes auto` would have picked without booting
the full stitching pipeline.

Usage:
    python tools/suggest_fg_classes_v2.py --video videos/sf_left.mp4

    # Or pipe into the stitcher:
    classes=$(python tools/suggest_fg_classes_v2.py --video videos/sf_left.mp4)
    python video_stitcher_seam_gpu.py --yoloe_fg_classes $classes ...

The detailed inventory + depth records + raw VLM response are
printed to stderr so stdout stays clean for piping.
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from stitcher.auto_fg_v2 import (  # noqa: E402
    read_frame_zero,
    suggest_fg_classes_v2,
)
from stitcher.vlm_judge import DEFAULT_OLLAMA_MODEL  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Print the auto-FG-v2 class list (inventory + "
                    "depth + VLM judge) for frame 0 of a video."
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
    parser.add_argument("--separator", default=", ",
                        help="String between class names on stdout. "
                             "Default: ', '. Pass ' ' for shell-friendly "
                             "(suitable for direct piping into "
                             "--yoloe_fg_classes).")
    args = parser.parse_args()

    print(f"[suggest_fg_classes_v2] reading frame 0 from {args.video}",
          file=sys.stderr)
    frame = read_frame_zero(args.video)
    H, W = frame.shape[:2]
    print(f"[suggest_fg_classes_v2] frame shape: H={H} W={W}",
          file=sys.stderr)
    print("[suggest_fg_classes_v2] running inventory + depth + VLM judge "
          f"(ollama: {args.ollama_model})...", file=sys.stderr)

    classes, records, raw = suggest_fg_classes_v2(
        frame,
        yoloe_weights=args.yoloe_weights,
        device=args.device,
        ollama_model=args.ollama_model,
        min_confidence=args.min_confidence,
        return_details=True,
    )

    # Detailed log -> stderr, clean class list -> stdout.
    print(f"[suggest_fg_classes_v2] {len(records)} detection(s):",
          file=sys.stderr)
    for r in records:
        print(f"  d={r['depth']:.2f}  conf={r['conf']:.2f}  "
              f"{r['class']}", file=sys.stderr)
    print("[suggest_fg_classes_v2] raw VLM response:", file=sys.stderr)
    print("  " + raw.replace("\n", "\n  "), file=sys.stderr)
    print(f"[suggest_fg_classes_v2] selected ({len(classes)}):",
          file=sys.stderr)

    sys.stdout.write(args.separator.join(classes))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
