# VideoStitcher

Real-time video stitching from two fixed cameras. Two synchronized
input streams (left + right) are warped onto a shared panorama canvas,
seam-stitched with awareness of moving people and static foreground
objects, and blended with a multi-band Laplacian pyramid so the seam is
invisible on the background.

Runs end-to-end on GPU (PyTorch grid_sample / conv2d / max_pool2d) when
CUDA is available; transparently falls back to a pure OpenCV / numpy
implementation when it isn't.

## What it does

- **Single fixed homography** estimated from the first frame pair (ORB
  + RANSAC); reused for the whole video.
- **Per-frame seam placement** via dynamic programming on a photometric
  cost map (smoothed across frames with an EMA), with hard penalties
  to keep the seam from crossing people or static foreground objects.
- **YOLO segmentation** for both people and foreground objects, with a
  choice of YOLOv8 (fast, fixed COCO classes) or YOLOE (open-vocabulary,
  text-prompted; slower but more accurate). Tasks pick independently
  via `--person_model` and `--fg_model`.
- **Multi-band Laplacian blending** around the seam for invisible
  transitions on the background.
- **FPS-desync correction**: if the two input streams have different
  nominal FPS, the slower one drives the pipeline and the faster one
  drops frames to stay temporally aligned.
- **Autocrop** of the polygonal stitched canvas to a clean rectangle.

## Quick start

### Install

```bash
pip install opencv-python numpy torch ultralytics
```

A CUDA-capable GPU is recommended (the YOLO models + grid_sample warp
benefit from it). The pipeline runs on CPU otherwise — slower, but
fully functional.

YOLO weights are auto-downloaded on first run:
- `yolov8n-seg.pt` (~7 MB, default for `--person_model yolov8` / `--fg_model yolov8`)
- `yoloe-11s-seg.pt` (~50 MB, default for `--person_model yoloe` / `--fg_model yoloe`)

### Run

```bash
python video_stitcher_seam_gpu.py \
    --video_a path/to/camA.mp4 \
    --video_b path/to/camB.mp4 \
    --output path/to/stitched.mp4
```

For a first run on a new scene, useful flags:

```bash
python video_stitcher_seam_gpu.py \
    --video_a A.mp4 --video_b B.mp4 --output out.mp4 \
    --autocrop --debug_seam --debug_mask \
    --max_frames 300
```

`--debug_seam` and `--debug_mask` overlay the DP seam (red line) and
the person/FG masks (red/yellow translucent) on the output so you can
inspect what's happening. `--max_frames 300` keeps the run short.

### Speed vs accuracy preset

Default is YOLOE for both person and FG detection (most accurate, ~2-3x
slower than YOLOv8). For maximum speed:

```bash
python video_stitcher_seam_gpu.py \
    --video_a A.mp4 --video_b B.mp4 --output out.mp4 \
    --person_model yolov8 --fg_model yolov8 --mask_ema 0.3
```

The `--mask_ema 0.3` enables temporal smoothing on the person mask,
which compensates for YOLOv8's higher per-frame jitter compared to
YOLOE.

## Full flag list

Run `python video_stitcher_seam_gpu.py --help` to see every flag.
The top-of-file docstring documents each flag with rationale, defaults,
and tuning guidance.

## Repository layout

```
video-stitcher/
├── video_stitcher_seam_gpu.py   ← entry point (CLI + main pipeline)
├── drafts/                      ← experiments that never made it into the pipeline
├── legacy/                      ← predecessor versions kept for reference
└── README.md
```

The current pipeline is `video_stitcher_seam_gpu.py`. Everything in
`legacy/` is archived (earlier CPU-only versions, motion-detection
experiments, stereo SGBM FG detection). `drafts/` contains exploratory
code that was tried and parked (SENA paper port, depth-based stitching,
manual correspondence picking).

## License

TBD.
