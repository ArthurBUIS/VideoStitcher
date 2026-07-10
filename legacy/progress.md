# legacy

## Purpose
Predecessor versions of the stitcher, kept for reference only. These are
the CPU-first / single-file ancestors that the current modular
`stitcher/` package + `video_stitcher_seam_gpu.py` grew out of. **None are
part of the live pipeline and none are imported by it.** They document the
evolution of the seam-avoidance idea (homography → multi-band → motion →
static-FG).

## Contents
| File | Stage in the lineage | Notes |
|------|----------------------|-------|
| `video_stitcher.py` | v1: fixed-camera homography + multi-band blend. | Homography via MAGSAC++ (`cv2.USAC_MAGSAC`), computed once from N calib frames and **saved/reloaded** from `homography.npy`. No seam DP, no masks. |
| `video_stitcher_v2.py` | v2: performance pass over v1. | Overlap-only pyramid blend, threaded warping, pipelined I/O, `xp` backend abstraction (numpy/cupy) toggled by `VIDEO_STITCH_DEVICE`. |
| `video_stitcher_seam.py` | v3: DP-seam + YOLO person mask. | Introduces the per-frame photometric cost map, EMA smoothing, DP seam, and person-avoidance — the design the current pipeline still uses. |
| `stitch_motion.py` | v4: adds motion avoidance. | MOG2 background subtraction (per-frame) unioned with the YOLO mask; person +1e8 / motion +5e7 split. MOG2 absorbs stopped objects over `--motion_history_seconds`. |
| `stitch_fg.py` | v5: adds static-foreground avoidance. | Static FG via **stereo disparity** from the overlap region (close objects = high disparity), thresholded into a parallax-risk mask, `--fg_recompute_seconds`. |

## Data flow
Each file is a self-contained single-file pipeline (read pair → warp →
[masks] → cost/seam or blend → write). The lineage is cumulative: each adds
one avoid-mask source. The current `stitcher/` package reimplements this
modularly and replaces the stereo-disparity static-FG detector
(`stitch_fg.py`) with YOLO segmentation + a per-detection depth filter, and
the MOG2 motion detector (`stitch_motion.py`) with empty-room
baseline subtraction.

## Dependencies
- **Internal:** none — no imports from `stitcher/`. Do not import these.
- **External:** `cv2`, `numpy`; optional `cupy` (v2's GPU backend),
  `ultralytics` (v3+ YOLO). Predate the PyTorch grid_sample GPU path.

## Key decisions & gotchas
- **Reference only; unmaintained.** Flags and behaviors differ from the
  current CLI (e.g. `--left/--right` here vs `--video_a/--video_b` now;
  `homography.npy` reload exists here but not in the live pipeline).
- Two abandoned approaches are recorded here vs their live replacements:
  stereo-disparity static FG (`stitch_fg.py`) → segmentation+depth
  (`stitcher/segmentation.py` + `static_fg.py`); MOG2 motion
  (`stitch_motion.py`) → baseline subtraction (`stitcher/motion.py`).

## Entry points
Each `.py` is an independent CLI script (see its docstring). Run directly;
none are imported.

## Pointers
- Parent / index: [../progress.md](../progress.md)
- Live successor: [../stitcher/progress.md](../stitcher/progress.md)
- Related experiments: [../drafts/progress.md](../drafts/progress.md)
