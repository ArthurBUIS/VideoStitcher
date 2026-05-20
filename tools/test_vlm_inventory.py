"""
Standalone validator for stitcher.vlm_inventory.list_objects_with_details.

Step 0 of auto-FG-v2: ask the VLM to list every object in the room,
with colors / materials / distinguishing details. Reads frame 0 of
a video, runs the VLM call, prints the raw response and the parsed
phrase list. No YOLOE or depth involved.

Usage:
    python tools/test_vlm_inventory.py --video videos/sf_left.mp4

Iterate on stitcher/vlm_inventory.py's INVENTORY_PROMPT until the
list matches what you would have catalogued by eye, then move on to
the chained auto-FG-v2 flow (Step 1+2+3 will run on this vocabulary).
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2  # noqa: E402

from stitcher.vlm_inventory import (  # noqa: E402
    DEFAULT_OLLAMA_MODEL,
    list_objects_with_details,
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


def main():
    parser = argparse.ArgumentParser(
        description="Run the VLM inventory step on frame 0 of a "
                    "video; print the parsed phrase list + raw "
                    "VLM response."
    )
    parser.add_argument("--video", required=True,
                        help="Path to a video; frame 0 is read.")
    parser.add_argument("--ollama_model", default=DEFAULT_OLLAMA_MODEL,
                        help="Ollama model tag.")
    args = parser.parse_args()

    print(f"[test_vlm_inventory] reading frame 0 from {args.video}")
    frame = _grab_frame_zero(args.video)
    H, W = frame.shape[:2]
    print(f"[test_vlm_inventory] frame shape: H={H} W={W}")
    print(f"[test_vlm_inventory] querying VLM ({args.ollama_model})...")

    phrases, raw = list_objects_with_details(
        frame, model_name=args.ollama_model, return_raw=True,
    )

    print("[test_vlm_inventory] raw VLM response:")
    print("  " + raw.replace("\n", "\n  "))
    print(f"[test_vlm_inventory] parsed inventory ({len(phrases)} phrases):")
    for p in phrases:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
