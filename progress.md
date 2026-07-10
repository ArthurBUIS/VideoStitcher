# VideoStitcher — repo map

## Overview
VideoStitcher is a real-time, GPU-accelerated pipeline that stitches two
synchronized **fixed-camera** video streams (left + right) into a single
seam-stitched panorama. A single homography is estimated once from the
first frame pair and reused for the whole video; each frame is then warped
to a shared canvas, a minimum-cost seam is placed by dynamic programming on
a photometric cost map (with hard penalties that keep the seam off moving
people, static foreground objects, and anything different from an
empty-room baseline), and the two halves are fused with a multi-band
Laplacian blend so the seam is invisible on the background. It runs
end-to-end on GPU via PyTorch (`grid_sample` / `conv2d` / `max_pool2d`) and
transparently falls back to a pure OpenCV/numpy path when CUDA is absent.

## Architecture
```
                      video_stitcher_seam_gpu.py   (CLI / argparse only)
                                   │  run(args)
                                   ▼
        ┌──────────────────────  stitcher.pipeline  ──────────────────────┐
        │  setup: device ▸ geometry ▸ warp grids ▸ gain ▸ segmenters ▸    │
        │         static-FG ▸ motion baselines ▸ writer                    │
        │                                                                  │
        │  threads (per frame, pipelined):                                 │
        │                                                                  │
        │   PrefetchingFrameReader ──► compute_worker ──► composite_worker │
        │      (io_utils, decode)      (compute_one)       (composite_one) │
        │                                  │  │                  │         │
        │                    yolo_q ◄──────┘  └──────► motion_q   ▼        │
        │                      │                          │   ThreadedVideoWriter
        │                 yolo_worker                 motion_worker  (io_utils)
        │              (segmentation)                   (motion)           │
        └──────────────────────────────────────────────────────────────────┘

  per-frame module use:
    geometry ─(startup)→ remap tables, overlap bbox, autocrop rect
    warp      → warp_pair_gpu (grid_sample) / cv2.remap  + gain
    segmentation → person mask (YOLO/YOLOE), static-FG mask ──uses──► depth_estimation, static_fg
    motion    → baseline-subtraction mask (pixel/edges/chrominance)
    seam      → cost + EMA + penalties → find_dp_seam (numba)
    compositing → multi-band Laplacian blend around the seam
    person_tracking → optional 3:2 rail crop on the closest person
```
Penalty priority in the cost map (additive): photometric <
`fg_penalty` = `motion_penalty` (5e7, where mask **and not** person) <
`person_penalty` (1e8).

## Directory index
| Directory | progress.md | One-line summary |
|-----------|-------------|------------------|
| `stitcher/` | [stitcher/progress.md](stitcher/progress.md) | **The live pipeline** — all setup + per-frame warp/mask/seam/blend logic across 14 modules. |
| `tools/` | [tools/progress.md](tools/progress.md) | Standalone validator for the depth-aware static-FG selector (per-detection keep/drop preview). |
| `drafts/` | [drafts/progress.md](drafts/progress.md) | Parked experiments (stereo/depth stitching, SENA port, manual pose) — not in the pipeline. |
| `legacy/` | [legacy/progress.md](legacy/progress.md) | Archived single-file predecessors (v1→v5) kept for reference. |

Root files (no directory of their own):
- `video_stitcher_seam_gpu.py` — **entry point.** Owns the full argparse
  (every CLI flag + its docstring rationale) and calls
  `stitcher.pipeline.run`.
- `README.md` — user-facing quick start + flag overview.

## Start here
| If you want to… | Read |
|-----------------|------|
| Understand or change the running pipeline | [stitcher/progress.md](stitcher/progress.md), then `stitcher/pipeline.py` (`run` + `compute_one`/`composite_one`). |
| Add/rename a CLI flag | `video_stitcher_seam_gpu.py` (argparse) → thread it through `stitcher/pipeline.py`. |
| Tune which objects the seam avoids | `stitcher/static_fg.py` (`ALWAYS_KEEP`/`FOREGROUND_ONLY`/`FG_DEPTH_THRESHOLD`); preview with [tools/progress.md](tools/progress.md). |
| Change seam placement / cost | `stitcher/seam.py`. |
| Change blending / seam visibility | `stitcher/compositing.py`. |
| Change motion detection | `stitcher/motion.py` (+ motion flags). |
| Understand geometry/homography/autocrop | `stitcher/geometry.py`. |
| See what was tried and abandoned | [drafts/progress.md](drafts/progress.md), [legacy/progress.md](legacy/progress.md). |

## Global conventions
- **Fixed cameras only.** Homography is computed once (frame 0) and reused;
  camera motion breaks alignment. Saved to `homography.npy` (written every
  run, not reloaded).
- **Camera A = left = warp reference**, B = right (warped into A's frame).
- **GPU/CPU parity:** most stages have a `_gpu` and `_cpu` variant selected
  by `device.detect_device()`; behavior should match, only the backend
  differs.
- **Per-frame work is restricted to the overlap bbox** (and motion to half
  that resolution) for speed.
- **Defaults:** `--person_model yoloe` and `--fg_model yoloe`; motion **on**
  (disable with `--no_motion`). YOLOE-FG vocab comes from `static_fg.py`,
  not the CLI.
- Progress docs follow the fixed section template (Purpose / Contents /
  Data flow / Dependencies / Key decisions & gotchas / Entry points /
  Pointers).

## Build / run
```bash
pip install opencv-python numpy torch ultralytics
# optional: numba (seam DP speedup), transformers pillow (depth filter)

python video_stitcher_seam_gpu.py \
    --video_a camA.mp4 --video_b camB.mp4 --output stitched.mp4

# first run on a new scene — inspect seam/masks/crop quickly:
python video_stitcher_seam_gpu.py --video_a A.mp4 --video_b B.mp4 \
    --output out.mp4 --autocrop --debug_seam --debug_mask --max_frames 300

# max speed preset:
python video_stitcher_seam_gpu.py ... \
    --person_model yolov8 --fg_model yolov8 --mask_ema 0.3
```
YOLO weights (`yolov8n-seg.pt` ~7 MB, `yoloe-11s-seg.pt` ~50 MB) auto-download
on first run. Run `python video_stitcher_seam_gpu.py --help` for the full
flag list; the entry-script module docstring documents every flag.

## Hardware requirements
- CUDA-capable GPU recommended (YOLO + grid_sample warp + pyramid blend all
  benefit). The depth-filter cost note in `stitcher/static_fg.py` was
  measured on a T1000. Runs fully on CPU otherwise — slower but functional.
- No specific CUDA/driver/PyTorch versions are pinned in-repo. `TODO(verify)`
  exact CUDA toolkit / driver / Python versions (no `requirements.txt`,
  `pyproject.toml`, or environment file is committed).

## TODO(verify)
- Exact dependency versions and Python version — no lockfile/manifest exists.
- License — `README.md` states "TBD".
