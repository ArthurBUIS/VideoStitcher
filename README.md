# VideoStitcher

Real-time video stitching from **two fixed cameras**. Two synchronized
input streams (left + right) are warped onto a shared panorama canvas,
seam-stitched with awareness of moving people and static foreground
objects, and blended with a multi-band Laplacian pyramid so the seam is
invisible on the background.

Runs end-to-end on GPU (PyTorch `grid_sample` / `conv2d` / `max_pool2d`)
when CUDA is available; transparently falls back to a pure OpenCV / numpy
implementation when it isn't.

```
  camera A (left) ──┐                                    ┌── stitched.mp4
                    ├─► warp ─► masks ─► seam ─► blend ──┤
  camera B (right) ─┘                                    └── (optional crop)
```

---

## Table of contents

- [The core idea](#the-core-idea)
- [What it does](#what-it-does)
- [Install](#install)
- [Run](#run)
- [Repository layout](#repository-layout)
- [Documentation map](#documentation-map)
- [Common tasks](#common-tasks)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)
- [License](#license)

---

## The core idea

Classic panorama stitching blends the two images across the whole overlap
region. That works for static scenes, but if a person is standing in the
overlap, they appear in *both* cameras at slightly different positions
(parallax) and blending produces a ghost.

VideoStitcher instead picks a **cut line** (the "seam") through the overlap
and takes pixels from camera A on one side and camera B on the other. The
seam is chosen every frame by dynamic programming over a cost map, so it
naturally routes through regions where the two cameras agree — and is
*forbidden* from crossing people or foreground objects via large additive
penalties. Blending is then applied only in a narrow band around the seam,
purely to hide exposure differences.

The three ideas that make it work:

1. **Fixed cameras ⇒ one homography.** The geometry is estimated once from
   the first frame pair (ORB + RANSAC) and reused for the whole video. All
   the expensive geometry (remap tables, validity masks, overlap bbox) is
   precomputed at startup.
2. **Seam placement is a cost-minimization.** Photometric disagreement
   between the two warped frames is the base cost; masks add penalties.
   The penalty hierarchy is additive and strictly ordered:

   | Source | Penalty | Meaning |
   |---|---|---|
   | Photometric disagreement | 0 – 1e5 | prefer regions where A and B agree |
   | Edge margin (`--edge_penalty`) | 1e6 | keep the seam away from canvas edges |
   | Static FG / motion (`--fg_penalty`, `--motion_penalty`) | 5e7 | detour around furniture and moved objects |
   | Person (`--person_penalty`) | 1e8 | **never** cut through a person |

   So the seam will cross a chair if that's the only way to avoid a person,
   but it will never cross a person to avoid a chair.
3. **Three independent "avoid" mask sources**, because no single one is
   sufficient:
   - **YOLO person segmentation** — catches people (the main ghosting risk).
   - **YOLO object segmentation + depth filter** — catches static
     foreground furniture, but only instances that are actually *near* the
     camera (a chair against the far wall has no parallax, so it's ignored).
   - **Baseline subtraction** — catches everything semantic segmentation
     misses: a moved chair, a dropped bag, an opened door. Anything that
     differs from a pre-recorded empty-room baseline.

---

## What it does

- **Single fixed homography** estimated from the first frame pair (ORB +
  RANSAC); reused for the whole video.
- **Per-frame seam placement** via dynamic programming on a photometric
  cost map (smoothed across frames with an EMA), with hard penalties to
  keep the seam off people and static foreground objects.
- **YOLO segmentation** for both people and foreground objects, with a
  choice of YOLOv8 (fast, fixed COCO classes) or YOLOE (open-vocabulary,
  text-prompted; slower but more accurate). The two tasks pick
  independently via `--person_model` and `--fg_model`.
- **Per-detection depth filtering** (YOLOE FG path): each detected object
  is kept or dropped by its bbox-median depth from Depth Anything V2, so
  two chairs in one frame get independent verdicts.
- **Motion detection via baseline subtraction** (on by default; disable
  with `--no_motion`) with a slow rolling baseline update, so the scene
  can drift without the mask firing forever.
- **Multi-band Laplacian blending** around the seam for invisible
  transitions on the background. Alternatives: `--naive_alpha_blend`
  (single-band feather) or `--no_blending` (hard cut, for inspecting seam
  placement).
- **FPS-desync correction**: if the two input streams have different
  nominal FPS, the slower one drives the pipeline and the faster one drops
  frames to stay temporally aligned. No frame is ever duplicated.
- **Autocrop** of the polygonal stitched canvas to a clean rectangle.
- **Optional person tracking** (`--person_tracking`): emit a 3:2 sub-crop
  that glides horizontally to follow the closest person.
- **Pipelined threading**: decode, compute, segmentation, motion, and
  encode each run on their own thread (and their own CUDA stream on GPU).

---

## Install

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install opencv-python numpy torch ultralytics
pip install numba transformers pillow   # optional, see below
```

**Required:** `opencv-python`, `numpy`, `torch`, `ultralytics`.

**Optional but recommended:**

| Package | Needed for | If missing |
|---|---|---|
| `numba` | seam DP fast path | falls back to numpy (~5× slower DP) |
| `transformers` + `pillow` | per-detection depth filter (`--fg_model yoloe`) | depth filter is skipped |

A CUDA-capable GPU is strongly recommended — the YOLO models, the
`grid_sample` warp, and the pyramid blend all benefit. The pipeline runs
fully on CPU otherwise, just much slower. Device selection is automatic
(`stitcher/device.py`); there is no flag to force one.

For a CUDA build of PyTorch, install it from
[pytorch.org](https://pytorch.org/get-started/locally/) rather than plain
`pip install torch`.

YOLO weights are auto-downloaded on first run:

- `yolov8n-seg.pt` (~7 MB) — used when `--person_model yolov8` or `--fg_model yolov8`
- `yoloe-11s-seg.pt` (~50 MB) — used when `--person_model yoloe` or `--fg_model yoloe` (both are the default)

---

## Run

Minimal invocation:

```bash
python video_stitcher_seam_gpu.py --video_a camA.mp4 --video_b camB.mp4 --output stitched.mp4
```

**Camera A is the left camera and is the warp reference**; B is the right
camera and is warped into A's frame. Swapping them will not work.

### First run on a new scene

Start short and with the debug overlays on, so you can see what the seam
and the masks are actually doing before committing to a full render:

```bash
python video_stitcher_seam_gpu.py --video_a A.mp4 --video_b B.mp4 --output out.mp4 --autocrop --debug_seam --debug_mask --max_frames 300
```

- `--debug_seam` draws the DP seam as a red line.
- `--debug_mask` overlays the person mask (red) and the static FG mask
  (yellow) as translucent layers.
- `--max_frames 300` keeps the run to ~10 seconds of footage.
- `--autocrop` removes the polygonal black borders of the raw canvas.

### Speed vs. accuracy

The default (`yoloe` for both tasks) is the most accurate configuration and
roughly 2–3× slower than YOLOv8. For maximum speed:

```bash
python video_stitcher_seam_gpu.py --video_a A.mp4 --video_b B.mp4 --output out.mp4 --person_model yolov8 --fg_model yolov8 --mask_ema 0.3
```

`--mask_ema 0.3` adds temporal smoothing to the person mask, compensating
for YOLOv8's higher per-frame jitter compared to YOLOE. Note that switching
`--fg_model` to `yolov8` also disables the depth filter (it only exists on
the YOLOE path) and falls back to fixed COCO class IDs from `--fg_classes`.

Other speed levers: raise `--yolo_every` (default 8), raise
`--seam_downscale` (default 4), lower `--blend_levels` (default 3), or
`--no_motion`.

### Motion baselines

Motion detection is **on by default**. If you have empty-room stills for
each camera, pass them — the results are much better than the fallback:

```bash
python video_stitcher_seam_gpu.py ... --motion_baseline_a empty_A.png --motion_baseline_b empty_B.png
```

Both must be given together, and each must match its camera's video
resolution. If omitted, frame 0 of each video is used as the baseline —
which means anything present in frame 0 is treated as "background".

### All flags

```bash
python video_stitcher_seam_gpu.py --help
```

The module docstring at the top of
[video_stitcher_seam_gpu.py](video_stitcher_seam_gpu.py) documents every
flag with its rationale, default, and tuning guidance. It is the
authoritative flag reference — this README only covers the common ones.

---

## Repository layout

```
VideoStitcher/
├── video_stitcher_seam_gpu.py   ← entry point: argparse only, calls stitcher.pipeline.run
├── stitcher/                    ← THE LIVE PIPELINE (14 modules)
│   ├── pipeline.py                orchestration: setup + worker threads  (the driver)
│   ├── device.py                  CUDA probe → per-stage device routing
│   ├── geometry.py                homography, canvas, remap tables, autocrop
│   ├── warp.py                    per-frame warp + gain compensation
│   ├── sync_reader.py             FPS-desync-aware paired frame reader
│   ├── io_utils.py                threaded decode / threaded writer / debug overlays
│   ├── segmentation.py            YOLOv8 + YOLOE wrapper, person & FG masks
│   ├── static_fg.py               ← editable FG vocabulary + depth threshold
│   ├── depth_estimation.py        Depth Anything V2 wrapper
│   ├── motion.py                  baseline-subtraction motion mask
│   ├── seam.py                    cost map, penalties, DP seam finder
│   ├── compositing.py             multi-band Laplacian blend
│   └── person_tracking.py         optional 3:2 rail crop on the closest person
├── tools/
│   └── test_static_fg.py        ← standalone previewer for the FG keep/drop filter
├── drafts/                      ← parked experiments (NOT in the pipeline)
├── legacy/                      ← archived single-file predecessors v1→v5
├── requirements.txt
└── README.md
```

`stitcher/` is the only code that runs in production. `drafts/` and
`legacy/` are reference material — nothing imports them, and their flags
and formats are stale. See their `progress.md` files for what was tried and
why it was dropped.

---

## Documentation map

Every directory carries a `progress.md` that follows a fixed template
(Purpose / Contents / Data flow / Dependencies / Key decisions & gotchas /
Entry points / Pointers). **These are the primary technical docs** — this
README is only the entry ramp.

| Doc | Covers |
|---|---|
| [progress.md](progress.md) | Repo index: architecture diagram, directory table, "start here" routing table, global conventions. **Read this first.** |
| [stitcher/progress.md](stitcher/progress.md) | Per-module responsibilities, key symbols, the full startup + per-frame data flow, and the threading/GPU-stream design. The most important document in the repo. |
| [tools/progress.md](tools/progress.md) | The static-FG validator. |
| [drafts/progress.md](drafts/progress.md) | Parked experiments and why they were parked. |
| [legacy/progress.md](legacy/progress.md) | The v1→v5 lineage and what each version added. |

Suggested reading order for someone new to the codebase:

1. This README (the [core idea](#the-core-idea) section).
2. [progress.md](progress.md) — architecture diagram and conventions.
3. [stitcher/progress.md](stitcher/progress.md) — module map and data flow.
4. The module docstring of `video_stitcher_seam_gpu.py` — every flag explained.
5. `stitcher/pipeline.py`, specifically `run()` then `compute_one()` / `composite_one()`.

---

## Common tasks

| If you want to… | Go to |
|---|---|
| Change which objects the seam avoids | `stitcher/static_fg.py` — edit the `ALWAYS_KEEP` / `FOREGROUND_ONLY` lists. Preview the effect with `python tools/test_static_fg.py --video A.mp4`. |
| Tune the near/far cutoff for FG objects | `FG_DEPTH_THRESHOLD` in `stitcher/static_fg.py`, or `--static_fg_depth_threshold` at runtime. Depth is normalized to [0, 1] where 1.0 = closest. |
| Change seam placement or cost | `stitcher/seam.py`; the penalty flags are `--person_penalty`, `--fg_penalty`, `--motion_penalty`, `--edge_penalty`, `--seam_lambda`. |
| Change blending / seam visibility | `stitcher/compositing.py`; flags `--blend_width`, `--blend_levels`, `--naive_alpha_blend`, `--no_blending`. |
| Change motion detection | `stitcher/motion.py`; flags `--motion_method`, `--motion_threshold`, `--baseline_update_alpha`. |
| Understand geometry / homography / autocrop | `stitcher/geometry.py`. |
| Add or rename a CLI flag | Add it to argparse in `video_stitcher_seam_gpu.py`, document it in that file's module docstring, then thread it through `stitcher/pipeline.py`. |
| Find where the time goes | `--profile` (host-side stage timings, near-zero overhead) or `--profile_stages` (true per-stage GPU latency; serializes the pipeline, so measure throughput separately). |
| See what was tried and abandoned | [drafts/progress.md](drafts/progress.md), [legacy/progress.md](legacy/progress.md). |

---

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Output is misaligned / doubled everywhere | The cameras moved, or `--video_a` / `--video_b` are swapped. The homography is estimated once from frame 0; any camera motion breaks it silently for the rest of the video. |
| Homography estimation fails or looks wrong | Not enough texture in the overlap on frame 0. Re-shoot with more visual detail in the shared region, or start the clip at a frame that has it. |
| The seam cuts through a person | Increase `--mask_dilate` (default 15) to absorb the A/B parallax offset on the mask; check with `--debug_mask` that the mask actually covers them. If the mask is flickering, lower `--mask_ema` (e.g. 0.3) or lower `--yolo_every`. |
| A person ghosts / appears twice | Same as above — the seam is running between the two views of the person. |
| Ghosting on furniture | The object isn't being detected or is being dropped by the depth filter. Run `python tools/test_static_fg.py --video A.mp4` to see the per-detection keep/drop verdicts, then adjust `ALWAYS_KEEP` / `FOREGROUND_ONLY` / `FG_DEPTH_THRESHOLD` in `stitcher/static_fg.py`. |
| The seam jitters frame to frame | Raise `--seam_lambda` (default 8.0) to pin it harder to the previous frame's seam, or lower `--cost_ema` (default 0.4) for more temporal smoothing. Too high and the seam reacts sluggishly when someone walks in. |
| The motion mask covers everything | Camera auto-exposure/white-balance drift. Renormalization is on by default; try `--motion_method edges` (robust to brightness *and* colour drift) or raise `--motion_threshold`. Note that thresholds are on different scales per method: ~200 for `pixel`, ~50 for `edges`, ~10–20 for `chrominance`. |
| A visible colour step across the seam | Check `--no_gain_comp` isn't set; raise `--blend_levels` for wider low-frequency blending. |
| Blend artifacts at the canvas edges | `--seam_edge_margin` must be at least `--blend_width / 2` so the pyramid blur doesn't reach padded pixels. |
| Output is black at the borders | You want `--autocrop`. The raw stitched canvas is polygonal. |
| Slow | See [Speed vs. accuracy](#speed-vs-accuracy). Confirm CUDA is actually being used — the pipeline prints its device choice at startup. |

---

## Known limitations

Things a new maintainer should know before promising anything:

- **Fixed cameras only, and this is load-bearing.** One homography for the
  whole video. There is no re-estimation, no drift correction, and no
  warning if the cameras move — the output just degrades silently.
- **`homography.npy` is written every run but never read back.** It's a
  debugging artifact, not a cache. (Legacy `video_stitcher.py` did reload
  it; the current pipeline does not.)
- **Offline, not live.** "Real-time" here means the per-frame cost is low
  enough to keep up; the CLI reads files and writes an mp4. There is no
  camera-capture or streaming input path.
- **No test suite.** `tools/test_static_fg.py` is a visual validator for one
  component, not a regression test. Changes are verified by eye on sample
  footage.
- **`--yoloe_fg_classes` is currently dead.** With `--fg_model yoloe` the
  vocabulary comes from `stitcher/static_fg.py`; with `--fg_model yolov8`
  the classes come from `--fg_classes` (COCO IDs). The flag is read by
  argparse and never used. Edit `static_fg.py` instead.
- **The depth-filter verdict logic is duplicated** between
  `stitcher/segmentation.py` (`_depth_filter_keep_mask`) and
  `tools/test_static_fg.py`. Changing one requires changing the other or
  the previewer stops matching runtime behaviour.
- **No pinned environment.** `requirements.txt` lists the packages but no
  versions, and no CUDA/driver/Python combination is recorded. Timings
  quoted in the docs (e.g. ~600 ms per depth recompute at 1440p) were
  measured on an NVIDIA T1000.

