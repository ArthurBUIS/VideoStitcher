"""
Standalone validator for stitcher.depth_fg.estimate_depth.

Reads frame 0 of a video, estimates depth, prints stats, and saves
a colourised depth map next to the input video for visual
inspection. Use this once after pulling the depth_fg module to
confirm the depth model loads and produces sensible output before
wiring it into the larger pipeline.

Usage:
    python tools/test_depth.py --video videos/sf_left.mp4

Output:
    videos/sf_left_depth.png   (colourised depth visualisation;
                                brighter = closer to the camera)

Setup (one-time):
    pip install transformers
    # The first run also auto-downloads Depth Anything V2 Small
    # (~50 MB) from HuggingFace.
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from stitcher.depth_fg import estimate_depth  # noqa: E402


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
        description="Estimate depth on frame 0 of a video and print stats. "
                    "Saves a colourised depth map next to the input video.",
    )
    parser.add_argument("--video", required=True,
                        help="Path to a video; frame 0 is read.")
    parser.add_argument("--output", default=None,
                        help="Output PNG path for the colourised depth map. "
                             "Default: <video_basename>_depth.png next to "
                             "the input.")
    args = parser.parse_args()

    print(f"[test_depth] reading frame 0 from {args.video}")
    frame = _grab_frame_zero(args.video)
    H, W = frame.shape[:2]
    print(f"[test_depth] frame shape: H={H} W={W}")

    print("[test_depth] loading Depth Anything V2 Small (one-time, "
          "~50 MB if not cached)...")
    depth = estimate_depth(frame)
    print(f"[test_depth] depth shape: {depth.shape}  dtype: {depth.dtype}")
    print(f"[test_depth] depth stats: min={depth.min():.4f} "
          f"max={depth.max():.4f} median={float(np.median(depth)):.4f}")

    # Normalise to 0-255 and colourise for visualisation.
    span = float(depth.max() - depth.min()) + 1e-8
    d_norm = (depth - depth.min()) / span
    d_u8 = (d_norm * 255).astype(np.uint8)
    d_color = cv2.applyColorMap(d_u8, cv2.COLORMAP_INFERNO)

    out_path = args.output or (
        os.path.splitext(args.video)[0] + "_depth.png"
    )
    cv2.imwrite(out_path, d_color)
    print(f"[test_depth] saved colourised depth -> {out_path}")
    print("[test_depth] visual sanity check: brighter pixels = closer "
          "to the camera, darker pixels = farther away.")


if __name__ == "__main__":
    main()
