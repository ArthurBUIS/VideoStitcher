"""
Real-time video stitching from two fixed cameras (GPU pipeline + CPU fallback).

Stitches two video streams from physically-fixed cameras into a single
panorama, with seam placement aware of moving people (YOLO person mask)
and static foreground objects (YOLO class segmentation: chairs, couches,
desks, etc.) so the seam never crosses through them. Multi-band Laplacian
blending makes the seam invisible on the background.

Runs end-to-end on GPU (PyTorch grid_sample, conv2d, max_pool2d) when
CUDA is available; transparently falls back to a pure OpenCV/numpy
implementation when it isn't.

Pipeline
--------
Startup (runs once, on the first paired frame):
    1. Estimate a single homography (ORB + RANSAC).
    2. Compute the panorama canvas size, build pixel-level remap tables
       (or grid_sample tensors on GPU), and pre-build static geometry:
       per-camera validity masks, the overlap region, its bbox.
    3. (Optional) Per-channel BGR gain compensation from frame 0.
    4. (Optional) Compute a static foreground mask via YOLO segmentation
       on a configurable list of COCO classes — warped, unioned, dilated.
    5. (Optional) Compute an autocrop rectangle from the homography.

Per frame:
    1. FrameSyncReader pulls a paired frame, dropping frames from the
       faster stream when the two input FPS values differ.
    2. Warp both frames to the canvas (gain folded in on GPU).
    3. Run YOLO every N frames; warp + dilate the union → "person mask".
    4. Photometric cost (squared BGR diff) over the overlap bbox; smoothed
       across frames via EMA.
    5. Inject penalties: forbid edges, fg-mask, person-mask. Add a
       quadratic attractor toward the previous frame's seam.
    6. Find the minimum-cost seam by dynamic programming on a downscaled
       cost map.
    7. Build a soft mask from the seam and run multi-band Laplacian
       pyramid blending on the overlap bbox; hard-copy the rest.
    8. (Optional) Crop to the autocrop rectangle.
    9. Write the frame.

FPS desync (FrameSyncReader)
----------------------------
If the two input FPS values differ by more than 0.5%, the slower stream
becomes the "driver" (one frame per pipeline tick) and the faster stream
becomes the "follower" (advance to the closest-in-time frame, drop the
rest). Output FPS = slower input FPS. No frame is ever duplicated. If
the FPS values match within tolerance, this is a zero-overhead lockstep
read. This corrects nominal-rate mismatch but not intra-stream jitter
or wall-clock drift unrelated to FPS declarations.

Command-line arguments
----------------------
Required:
    --video_a PATH              Input video from camera A (left).
    --video_b PATH              Input video from camera B (right).
    --output PATH               Output stitched video (.mp4).

General:
    --max_frames N              Process only the first N frames (0 = all).
                                Useful for quick iteration. Default: 0.
    --debug_seam                Overlay the DP seam as a red line on the
                                output.
    --debug_mask                Overlay the person mask (red) and the
                                static FG mask (yellow) as translucent
                                overlays on the output.
    --autocrop                  Crop the output to a clean axis-aligned
                                rectangle. The right edge is set by the
                                more-conservative (smaller-x) of B's two
                                warped right corners; the left edge is
                                A's left edge on canvas; vertical extent
                                spans both right corners. Saves disk
                                space and removes the polygonal black
                                borders of the raw stitched canvas.

Segmentation models (one per task; share a model when both tasks pick the
same type):
    --person_model {yolov8,yoloe}
                                Model used for person detection. yoloe
                                is more accurate (esp. on edge cases like
                                partial occlusions) but ~2-3x slower than
                                yolov8. Default: yoloe.
    --fg_model {yolov8,yoloe}   Model used for static FG detection.
                                yoloe lets you target arbitrary object
                                types via text prompts; yolov8 is
                                limited to the 80 COCO classes. Default:
                                yoloe.
    --yolo_weights PATH         YOLOv8 weights file. Used when either
                                --person_model or --fg_model is yolov8.
                                Default: yolov8n-seg.pt.
    --yoloe_weights PATH        YOLOE weights file. Used when either
                                --person_model or --fg_model is yoloe.
                                Default: yoloe-11s-seg.pt.
    --yoloe_person_class STR    Text prompt for the person class when
                                --person_model is yoloe. Default:
                                "person".
    --yoloe_fg_classes STR ...  Text prompts for static FG classes when
                                --fg_model is yoloe. Multi-word prompts
                                must be quoted ("dining table"). Default:
                                chair couch bed "dining table" tv laptop
                                book "potted plant" backpack.

Person mask:
    --yolo_every N              Run the person model once every N frames;
                                reuse the cached mask in between. Lower =
                                fresher mask but slower. Default: 5.
    --mask_dilate PX            Dilation radius applied to the unioned
                                person mask, in pixels. Absorbs the
                                parallax offset between A and B's view of
                                the same person. Increase if the seam
                                grazes a person's outline; decrease if
                                the mask engulfs background. Default: 15.
    --mask_ema A                EMA factor in [0, 1] applied to the
                                person mask between consecutive runs.
                                Lower = more temporal smoothing (less
                                jitter, slower to react to genuine
                                motion). 1.0 disables smoothing.
                                Default: 1.0.
                                Mainly useful with --person_model yolov8,
                                which is jitterier than yoloe; yoloe's
                                masks are usually stable enough that
                                smoothing isn't needed.
    --mask_ema_threshold T      Threshold applied to the smoothed (EMA)
                                person mask to obtain the binary mask
                                used by the cost map. Ignored when
                                --mask_ema is 1.0. Default: 0.6.

Static foreground (segmentation-based):
    --no_fg                     Disable static FG detection entirely.
    --fg_classes CLASS_IDS ...  Space-separated COCO class IDs to treat
                                as static foreground. Used only when
                                --fg_model is yolov8. Default: 56 57 59
                                60 62 63 73 (chair, couch, bed, dining
                                table, tv, laptop, book). For yoloe, see
                                --yoloe_fg_classes.
    --fg_dilate PX              Dilation radius for the FG mask, in
                                pixels. Default: 10.
    --fg_recompute_seconds F    Seconds between FG mask recomputations.
                                0 = compute once at startup and never
                                again. Increase if the scene has
                                furniture that gets rearranged during
                                the recording. Default: 0.

Cost-map behavior:
    --cost_ema A                EMA factor in [0, 1] for the photometric
                                cost. Lower = smoother but slower to
                                react. Higher = more reactive but
                                jitterier. Default: 0.4.
    --no_cost_ema               Disable EMA entirely (equivalent to
                                cost_ema = 1.0).
    --seam_lambda F             Strength of the quadratic "stay near the
                                previous seam" attractor. 0 disables it.
                                Higher pins the seam harder; too high
                                and the seam reacts sluggishly when a
                                person approaches. Default: 8.0.
    --seam_edge_margin N        Width in pixels of the forbidden band at
                                the left/right edges of the overlap
                                bbox. Should be at least blend_width / 2
                                so the multi-band blur doesn't reach
                                into padded pixels. 0 disables.
                                Default: 50.
    --edge_penalty F            Cost added to pixels inside the
                                seam_edge_margin band. Default: 1e6.

Crossing penalties (added to the cost map at seam-finding time):
    --person_penalty F          Cost added to pixels covered by the
                                YOLO person mask (highest priority —
                                wins over FG when they overlap).
                                Default: 1e8.
    --fg_penalty F              Cost added to FG-AND-NOT-person pixels.
                                Default: 5e7.

  The intended hierarchy is fg_penalty < person_penalty so that the
  seam will detour around static FG when it can, but is forbidden from
  cutting through people even at the cost of crossing FG.

Seam computation:
    --seam_downscale N          Factor by which the cost map is
                                downscaled before DP. Higher = much
                                faster DP, but coarser seam. Default: 4.

Gain compensation:
    --no_gain_comp              Disable global per-channel gain
                                compensation. With multi-band blending
                                on, gain comp is partially redundant —
                                the coarsest pyramid band already
                                handles low-frequency exposure
                                matching — but disabling it can leave a
                                faint colour step depending on the
                                cameras.

Multi-band blending:
    --blend_width PX            Width in pixels of the soft mask ramp
                                around the DP seam. Default: 80.
                                Constraint: seam_edge_margin >=
                                blend_width / 2.
    --blend_levels N            Laplacian pyramid depth. Higher = wider
                                low-frequency blending (better exposure
                                hiding) at the cost of more pyrDown /
                                pyrUp per frame. Default: 3.

Usage
-----
    python video_stitcher_seam_gpu.py \\
        --video_a camA.mp4 --video_b camB.mp4 --output stitched.mp4

The default uses YOLOE for both person and FG detection (highest
accuracy, ~2-3x slower than YOLOv8). For maximum speed, switch both
back to YOLOv8 and enable temporal smoothing:

    python video_stitcher_seam_gpu.py ... \\
        --person_model yolov8 --fg_model yolov8 --mask_ema 0.3

For a first run on a new scene, add --debug_seam --debug_mask --autocrop
and reduce --max_frames to inspect the seam, the masks, and the crop
rectangle quickly.
"""

import argparse

from stitcher.pipeline import run
from stitcher.seam import EDGE_PENALTY, PERSON_PENALTY
from stitcher.segmentation import DEFAULT_FG_CLASS_IDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_a", required=True)
    parser.add_argument("--video_b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--debug_seam", action="store_true")
    parser.add_argument("--debug_mask", action="store_true")
    parser.add_argument("--autocrop", action="store_true",
                        help="Crop output to the largest axis-aligned "
                             "rectangle inside the stitched canvas.")
    parser.add_argument("--yolo_every", type=int, default=5)
    parser.add_argument("--mask_dilate", type=int, default=15)
    parser.add_argument("--mask_ema", type=float, default=1.0,
                        help="EMA factor in [0, 1] for the person mask. "
                             "Lower = more temporal smoothing (less "
                             "jitter, slower to react). 1.0 disables. "
                             "Mainly useful with --person_model yolov8, "
                             "which is jitterier than yoloe. Default: 1.0.")
    parser.add_argument("--mask_ema_threshold", type=float, default=0.6,
                        help="Threshold applied to the smoothed person "
                             "mask to obtain the binary mask used for "
                             "the cost map. Ignored when --mask_ema is "
                             "1.0. Default: 0.6.")
    parser.add_argument("--seam_downscale", type=int, default=4)
    # Segmentation model selection ----------------------------------------
    parser.add_argument("--person_model", choices=["yolov8", "yoloe"],
                        default="yoloe",
                        help="Which model to use for person detection. "
                             "yoloe is more accurate but slower. "
                             "Default: yoloe.")
    parser.add_argument("--fg_model", choices=["yolov8", "yoloe"],
                        default="yoloe",
                        help="Which model to use for static foreground "
                             "detection. yoloe lets you target arbitrary "
                             "object types via text prompts. Default: yoloe.")
    parser.add_argument("--yolo_weights", default="yolov8n-seg.pt",
                        help="YOLOv8 weights file. Default: yolov8n-seg.pt.")
    parser.add_argument("--yoloe_weights", default="yoloe-11s-seg.pt",
                        help="YOLOE weights file. Default: yoloe-11s-seg.pt.")
    parser.add_argument("--yoloe_person_class", default="person",
                        help="Text prompt for the person class when "
                             "--person_model is yoloe. Default: 'person'.")
    parser.add_argument("--yoloe_fg_classes", type=str, nargs="+",
                        default=["chair", "couch", "bed", "dining table",
                                 "tv", "laptop", "book", "potted plant",
                                 "backpack"],
                        help="Text prompts for static FG classes when "
                             "--fg_model is yoloe. Default: chair couch "
                             "bed 'dining table' tv laptop book "
                             "'potted plant' backpack.")
    parser.add_argument("--no_gain_comp", action="store_true")
    parser.add_argument("--cost_ema", type=float, default=0.4)
    parser.add_argument("--no_cost_ema", action="store_true")
    parser.add_argument("--blend_width", type=int, default=80)
    parser.add_argument("--blend_levels", type=int, default=3)
    parser.add_argument("--seam_lambda", type=float, default=8.0)
    parser.add_argument("--seam_edge_margin", type=int, default=50)
    parser.add_argument("--person_penalty", type=float, default=PERSON_PENALTY,
                        help=f"Cost penalty for person-mask pixels "
                             f"(default {PERSON_PENALTY:g}).")
    parser.add_argument("--edge_penalty", type=float, default=EDGE_PENALTY,
                        help=f"Cost penalty for the seam_edge_margin band "
                             f"(default {EDGE_PENALTY:g}).")
    # Static foreground (segmentation-based) flags.
    parser.add_argument("--no_fg", action="store_true",
                        help="Disable static foreground detection.")
    parser.add_argument("--fg_classes", type=int, nargs="+",
                        default=DEFAULT_FG_CLASS_IDS,
                        help="COCO class IDs for static foreground "
                             "(default: chair, couch, bed, table, tv, "
                             "laptop, book).")
    parser.add_argument("--fg_dilate", type=int, default=10,
                        help="FG mask dilation radius in px (default 10).")
    parser.add_argument("--fg_penalty", type=float, default=5e7,
                        help="Cost penalty for FG pixels (default 5e7).")
    parser.add_argument("--fg_recompute_seconds", type=float, default=0.0,
                        help="Seconds between FG recomputations "
                             "(0 = startup only).")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
