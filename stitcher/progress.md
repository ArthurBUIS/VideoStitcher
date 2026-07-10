# stitcher

## Purpose
The core stitching package. Turns two synchronized fixed-camera video
streams into a single seam-stitched panorama, entirely on GPU (PyTorch)
when CUDA is available and on a numpy/OpenCV fallback otherwise. All
per-frame work — warp, person/FG/motion masks, seam-cost DP, multi-band
blend — lives here; the root entry script only parses CLI flags and calls
`pipeline.run(args)`.

## Contents
| File | Responsibility | Key symbols |
|------|----------------|-------------|
| `__init__.py` | Empty package marker (namespace only, 0 lines). | — |
| `pipeline.py` | End-to-end orchestration: setup + 5 worker threads + per-frame compute/composite closures. **The only stateful driver.** | `run`, `compute_one`, `composite_one`, `yolo_worker`, `motion_worker`, `compute_worker`, `composite_worker`, `_build_segmenters`, `StageTimer`, `HOMOGRAPHY_PATH` |
| `device.py` | Probe CUDA once; return per-stage device routing dict. | `detect_device` |
| `geometry.py` | Startup-only geometry: homography, canvas, remap tables, per-camera validity masks, overlap bbox, autocrop rect. | `estimate_homography`, `compute_canvas`, `find_autocrop_rect`, `build_remap`, `build_static_geometry` |
| `warp.py` | Per-frame warp + gain comp (GPU grid_sample / CPU LUT+remap), mask warp + separable dilation. | `warp_pair_gpu`, `warp_gpu`, `warp_mask_gpu`, `dilate_gpu`, `compute_gain_compensation`, `build_gain_tensor`, `build_gain_lut`, `apply_gain_lut`, `build_grid_sample_tensor` |
| `sync_reader.py` | FPS-desync-aware paired frame reader (slower stream drives, faster drops frames). | `FrameSyncReader`, `DESYNC_TOLERANCE` |
| `io_utils.py` | Threaded decode prefetch, threaded (sync + async) video writer, debug overlay drawers. | `PrefetchingFrameReader`, `ThreadedVideoWriter`, `draw_seam_overlay`, `draw_mask_overlay` |
| `segmentation.py` | YOLOv8 / YOLOE wrapper for person + static-FG masks; per-detection depth filter; canvas-warp + union + dilate helpers. | `PersonSegmenter`, `compute_fg_mask_seg_gpu`, `compute_fg_mask_seg_cpu`, `_depth_filter_keep_mask`, `PERSON_CLASS_ID`, `DEFAULT_FG_CLASS_IDS` |
| `static_fg.py` | Editable YOLOE vocab tiers (`ALWAYS_KEEP` / `FOREGROUND_ONLY`) + depth threshold for the runtime static-FG selector. | `ALWAYS_KEEP`, `FOREGROUND_ONLY`, `FG_DEPTH_THRESHOLD`, `get_combined_vocab`, `get_fg_only_indices` |
| `depth_estimation.py` | Depth Anything V2 (Small) via HF transformers; scene-relative depth normalization; bbox-median lookup. | `estimate_depth`, `normalize_depth`, `bbox_median_depth`, `release_depth` |
| `seam.py` | Photometric cost + EMA + penalty injection; DP seam finder (numba fast path); edge-margin + previous-seam regularizers. | `compute_cost_and_ema_gpu`, `compute_cost_fast_cpu`, `find_dp_seam`, `upscale_seam`, `add_edge_margin_penalty`, `add_seam_regularizer`, `PERSON_PENALTY`, `EDGE_PENALTY` |
| `compositing.py` | Multi-band Laplacian blend around the seam (GPU conv2d / CPU pyrDown-Up), strip-limited; soft alpha mask builder. | `composite_multiband_gpu_async`, `composite_multiband_cpu`, `build_soft_mask_fast`, `get_pyr_kernel_2d` |
| `motion.py` | Baseline-subtraction motion mask (pixel / edges / chrominance), half-res bbox helpers, per-frame renormalization. | `compute_motion_mask_gpu`/`_cpu` (+ `_edges`, `_chrominance`), `renormalize_to_baseline_gpu`/`_cpu`, `sobel_magnitude_*`, `precompute_baseline_ab_*`, `MOTION_DOWNSCALE` |
| `person_tracking.py` | Optional 3:2 horizontal-rail crop centred on the closest person; two-time-constant EMA. | `PersonTracker`, `_largest_blob_centroid_x` |

## Data flow
Inputs: two `cv2.VideoCapture` streams (paths from CLI). Output: one `.mp4`.

**Startup (once, in `pipeline.run`):**
1. `detect_device` → routing dict.
2. `FrameSyncReader.read()` → first paired frame → `estimate_homography`
   (ORB+RANSAC, B→A) → saved to `homography.npy`.
3. `compute_canvas` → canvas size + translation `T`; if `--autocrop`,
   `find_autocrop_rect` folds a crop translation into the homographies so
   the warp only ever produces the shipped pixels.
4. `build_remap` (map_x/map_y) + `build_static_geometry` (validity masks,
   overlap bbox). On GPU, `build_grid_sample_tensor` converts the maps to
   a grid_sample tensor.
5. `compute_gain_compensation` from frame 0 → gain tensor/LUT.
6. `_build_segmenters` → one/two `PersonSegmenter`s; YOLOE-FG vocab from
   `static_fg.get_combined_vocab`.
7. `compute_fg_mask_seg_gpu/cpu` → initial static-FG mask (with optional
   per-detection depth filter via `depth_estimation`).
8. Motion baselines warped, cropped to bbox, downsampled to half-res.

**Per frame (pipelined across threads):**
`prefetch_reader.read` (decode) → `compute_in_q` → **`compute_one`**:
`warp_pair_gpu` → submit to `motion_q` + `yolo_q` (async) → read latest
person/motion masks from holders → `compute_cost_and_ema_gpu` /
`compute_cost_fast_cpu` (photometric + EMA + fg/motion/person/edge
penalties) → `add_seam_regularizer` → `find_dp_seam` (on downscaled cost)
→ `upscale_seam` → payload. Then `composite_in_q` → **`composite_one`**:
`composite_multiband_gpu_async` / `composite_multiband_cpu` → debug
overlays + optional tracking crop → `ThreadedVideoWriter`.
Side threads: `yolo_worker` (owns person-mask EMA), `motion_worker` (owns
rolling baseline), both publish to lock-guarded holders read by
`compute_one` (1-frame lag tolerated).

## Dependencies
- **Internal (intra-package):** `pipeline` imports from every other module.
  `segmentation` → `warp` (dilate/warp mask) and lazily → `depth_estimation`
  + `static_fg`. `motion` → `warp` (`dilate_gpu`). `compositing`,
  `seam`, `person_tracking` are leaf modules. Relied on by: the root entry
  script `video_stitcher_seam_gpu.py` (imports `run`, `EDGE_PENALTY`,
  `PERSON_PENALTY`, `DEFAULT_FG_CLASS_IDS`) and `tools/test_static_fg.py`
  (imports `static_fg`, `depth_estimation`, `segmentation`).
- **External:** `torch` + `torch.nn.functional` (GPU path), `cv2`
  (opencv-python), `numpy`. Optional: `ultralytics` (`YOLO` / `YOLOE`,
  required to run at all), `numba` (seam DP ~5× speedup; falls back to
  numpy), `transformers` + `PIL`/Pillow (depth filter only, lazy-imported).
- **Hardware:** CUDA GPU strongly recommended; runs on CPU otherwise.
  Depth-filter cost note in `static_fg.py` references a T1000 GPU.

## Key decisions & gotchas
- **Fixed-camera assumption is load-bearing.** A single homography is
  estimated from frame 0 and reused for the entire video. Any camera motion
  breaks the alignment silently.
- **`homography.npy` is written every run (`HOMOGRAPHY_PATH`, cwd) but never
  reloaded** by this pipeline — it's an artifact, not a cache. (Legacy
  `video_stitcher.py` had reload; this one does not.)
- **Motion detection is ON by default.** CLI flags are inverted
  (`--no_motion`, `--no_motion_renorm`); `run` maps them back to positive
  `args.motion` / `args.motion_renorm` predicates (pipeline.py ~L537).
- **Penalty hierarchy is additive and ordered:** photometric (0–1e5) <
  `fg_penalty`/`motion_penalty` (5e7, applied where mask AND NOT person) <
  `person_penalty` (1e8). Person always wins overlaps.
- **Threading uses lock-guarded single-slot holders, not queues, for
  mask handoff.** `yolo_worker`/`motion_worker` publish the latest mask;
  `compute_one` reads whatever is current (person mask up to `--yolo_every`
  frames stale, motion mask 1 frame stale). GPU cross-stream sync is done
  with `torch.cuda.Event().wait()` (no host stall). Each critical worker
  runs on its own CUDA stream (compute/composite/yolo high priority, motion
  default) so kernels from consecutive frames interleave.
- **Async composite path** hands a pinned host buffer + a CUDA event to the
  writer thread; a ring of 10 pinned buffers provides backpressure.
- **Motion runs at half bbox resolution** (`MOTION_DOWNSCALE=2`); dilate
  radius is halved to keep the same effective footprint after upsample.
- **Rolling motion baseline is gated by person-mask ONLY, not motion mask**
  — motion-gating deadlocks (a baseline that captured content which later
  vanishes fires motion forever, blocking the update that would clear it).
  Baselines stored as float32 so the small per-frame `alpha` update
  (default 0.01) isn't truncated by a uint8 cast.
- **YOLOE-FG vocabulary comes from `static_fg.py`, not the CLI.**
  `--yoloe_fg_classes` is ignored when `--fg_model yoloe`; edit
  `ALWAYS_KEEP` / `FOREGROUND_ONLY` instead.
- **Depth filter is per-detection, not per-class:** two chairs from one
  YOLOE call get independent keep/drop verdicts by bbox-median depth.
- `find_autocrop_rect` inscribes a rectangle in the *rasterized union
  polygon* (findContours + approxPolyDP), which is stricter than using only
  the 8 input corners.

## Entry points
Not run directly. Imported by:
- `video_stitcher_seam_gpu.py` (repo root) → `from stitcher.pipeline import run`.
- `tools/test_static_fg.py` → `stitcher.static_fg`, `stitcher.depth_estimation`,
  `stitcher.segmentation`.
No unit-test suite for this package (only the standalone `tools/` validator).

## Pointers
- Parent / index: [../progress.md](../progress.md)
- Sibling dirs: [../tools/progress.md](../tools/progress.md) ·
  [../drafts/progress.md](../drafts/progress.md) ·
  [../legacy/progress.md](../legacy/progress.md)
