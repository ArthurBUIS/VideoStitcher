# drafts

## Purpose
Exploratory code that was tried and parked — **not part of the shipping
pipeline** and not imported by it. Kept for reference on approaches that
were investigated (depth/stereo stitching, a SENA-paper port, manual
correspondence-based pose recovery). Treat everything here as archived;
the live pipeline is `stitcher/` + `video_stitcher_seam_gpu.py`.

## Contents
| File | Responsibility | Key symbols / notes |
|------|----------------|---------------------|
| `calibrate.py` | Camera intrinsic calibration from a chessboard video (`cv2.calibrateCamera`); saves `K` + distortion to `.npz`. | CLI: `--video --rows --cols --square_size --output` |
| `pick_correspondences.py` | Interactive GUI to manually click matching static points between frame 0 of two videos; saves to JSON. | GUI controls (u/r/s/q/h); output `clicks.json` |
| `pose_from_clicks.py` | Recover relative camera pose from the manual clicks; sweeps focal length to minimize epipolar residual. | `epipolar_residual`; output `pose.npz` (K, R_ba, t_ba, focal) |
| `visualize_setup.py` | Matplotlib 3D sanity-check of recovered camera geometry (frustums + triangulated points). | `frustum_lines`; needs `matplotlib` |
| `depth_stitch.py` | Depth-aware stereo video stitching proof-of-concept (v3): focal sweep, translation injection, epipolar residual reporting. | CLI `--video_a/--video_b --force_translation --calib_a/_b` |
| `sena_core.py` | Port of the SENA stitching algorithm (Tchana et al., 2026): local-affine warp grid + FFD field, adequate-zone detector, anchor partitioner. | `LocalAffineWarper`, `AdequateZoneDetector`, `AnchorPartitioner` |
| `video_stitcher_SENA.py` | Fixed-camera pipeline built on `sena_core`. **Marked "Not working for the moment"** at the top of the file. | — |
| `sena_state.npz` | Saved SENA state artifact (binary; git-ignored patterns exist but this one is committed). | data only |

## Data flow
No single flow — these are independent experiments. The
calibrate → pick_correspondences → pose_from_clicks → visualize_setup →
depth_stitch chain forms one investigation (calibrated/stereo geometry);
sena_core → video_stitcher_SENA forms another (SENA warp). Neither chain
feeds the production pipeline.

## Dependencies
- **Internal:** none on `stitcher/` (self-contained scripts). Do not import
  these from production code.
- **External:** `cv2`, `numpy`; plus `matplotlib` (`visualize_setup`),
  XFeat + SIFT fallback (`sena_core`, per its docstring).

## Key decisions & gotchas
- **Nothing here is maintained against the current pipeline.** Symbols,
  flags, and formats may be stale.
- `video_stitcher_SENA.py` is explicitly non-functional.
- The stereo/depth line of work was superseded by the segmentation +
  per-detection depth-filter approach now in `stitcher/static_fg.py` and
  `stitcher/depth_estimation.py`.

## Entry points
Each `.py` is an independent CLI script (see its top-of-file docstring).
Run directly; none are imported.

## Pointers
- Parent / index: [../progress.md](../progress.md)
- Superseded-by (live equivalents): [../stitcher/progress.md](../stitcher/progress.md)
- See also archived predecessors: [../legacy/progress.md](../legacy/progress.md)
