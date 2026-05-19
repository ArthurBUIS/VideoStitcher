"""
Standalone driver for stitcher.auto_fg.suggest_fg_classes.

Reads frame 0 from each of the two input videos, asks a local VLM
(Qwen2.5-VL via Ollama by default) to list the static foreground
classes in the room, and prints the result as a comma-separated list
on stdout — suitable for piping straight into --yoloe_fg_classes.

Diagnostics go to stderr so stdout stays clean for piping.

Usage:
    python tools/suggest_fg_classes.py \\
        --video_a videos/cam_a.mp4 --video_b videos/cam_b.mp4

    # Pipe into the main pipeline:
    python video_stitcher_seam_gpu.py \\
        --video_a videos/cam_a.mp4 --video_b videos/cam_b.mp4 \\
        --output stitched.mp4 \\
        --yoloe_fg_classes $(python tools/suggest_fg_classes.py \\
            --video_a videos/cam_a.mp4 --video_b videos/cam_b.mp4)

    # Or run with the in-pipeline shortcut (calls this script's
    # function internally):
    python video_stitcher_seam_gpu.py ... --yoloe_fg_classes auto

Setup (one-time):
    1. Install Ollama: https://ollama.com
    2. Pull the model: ollama pull qwen2.5vl:3b
    3. pip install ollama
"""

import argparse
import os
import sys

# Allow `python tools/suggest_fg_classes.py` to find the stitcher
# package when invoked from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2  # noqa: E402

from stitcher.auto_fg import DEFAULT_OLLAMA_MODEL, suggest_fg_classes  # noqa: E402


def _grab_frame_zero(path):
    cap = cv2.VideoCapture(path)
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame 0 from {path}")
    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Discover the static-FG class list for a target "
                    "room by querying a local VLM. Prints a comma-"
                    "separated list to stdout."
    )
    parser.add_argument("--video_a", required=True,
                        help="Path to camera A video (frame 0 is read).")
    parser.add_argument("--video_b", required=True,
                        help="Path to camera B video (frame 0 is read).")
    parser.add_argument("--ollama_model", default=DEFAULT_OLLAMA_MODEL,
                        help=f"Ollama model tag to use. "
                             f"Default: {DEFAULT_OLLAMA_MODEL}.")
    args = parser.parse_args()

    print(f"[suggest_fg_classes] reading frame 0 from {args.video_a}",
          file=sys.stderr)
    frame_a = _grab_frame_zero(args.video_a)
    print(f"[suggest_fg_classes] reading frame 0 from {args.video_b}",
          file=sys.stderr)
    frame_b = _grab_frame_zero(args.video_b)

    print(f"[suggest_fg_classes] querying {args.ollama_model} "
          f"(can take 5-30s on a T1000 — first run is slower)",
          file=sys.stderr)
    classes = suggest_fg_classes(
        frame_a, frame_b, model_name=args.ollama_model,
    )

    print(f"[suggest_fg_classes] discovered ({len(classes)} classes): "
          f"{classes}", file=sys.stderr)

    # Clean CSV on stdout — pipe-friendly. Quoting wrapped each item so
    # multi-word classes like "yellow chair" survive shell tokenisation
    # when piped through `$( ... )` into --yoloe_fg_classes.
    print(" ".join(f'"{c}"' for c in classes))


if __name__ == "__main__":
    main()
