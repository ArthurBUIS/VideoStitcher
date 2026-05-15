"""
End-to-end stitching pipeline.

`run(args)` takes a parsed argparse.Namespace, sets up the device /
geometry / segmenters / writer, then runs the per-frame loop. The
entry point script (`video_stitcher_seam_gpu.py`) owns the argparse;
this module owns the actual work.
"""

import collections
import contextlib
import os
import queue
import threading
import time

# `with _nullcontext():` is a no-op context manager — used when CUDA isn't
# available so we can keep one code path for both GPU + CPU workers.
_nullcontext = contextlib.nullcontext


class StageTimer:
    """Single-writer / single-reader rolling timer for one pipeline stage.

    Each StageTimer is mutated by exactly one thread (the worker that
    owns it) and only read by the profile printer + the end-of-run
    summary, so we don't bother with locks — the small chance of a
    half-updated read is fine for diagnostic output.
    """

    def __init__(self, recent_len=200):
        self.count = 0
        self.total_ms = 0.0
        self.recent = collections.deque(maxlen=recent_len)

    def record(self, ms):
        self.count += 1
        self.total_ms += ms
        self.recent.append(ms)

    def summary(self):
        if self.count == 0:
            return "(no samples)"
        avg_all = self.total_ms / self.count
        if self.recent:
            r_arr = list(self.recent)
            r_avg = sum(r_arr) / len(r_arr)
            r_max = max(r_arr)
        else:
            r_avg = avg_all
            r_max = 0.0
        return (f"n={self.count:>6d}  avg={avg_all:6.2f}ms  "
                f"recent_avg={r_avg:6.2f}ms  recent_max={r_max:6.2f}ms")


def _make_profile():
    return {
        "decode":             StageTimer(),  # main: prefetch_reader.read()
        "main_put_wait":      StageTimer(),  # main: put → compute_in_q
        "compute_get_wait":   StageTimer(),  # compute: get ← compute_in_q
        "compute":            StageTimer(),  # compute: compute_one()
        "compute_put_wait":   StageTimer(),  # compute: put → composite_in_q
        "composite_get_wait": StageTimer(),  # composite: get ← composite_in_q
        "composite":          StageTimer(),  # composite: composite_one()
        "composite_write":    StageTimer(),  # composite: writer.write()
        "yolo_get_wait":      StageTimer(),  # yolo: get ← yolo_q
        "yolo":               StageTimer(),  # yolo: per-frame YOLO+post-proc
    }


def _print_profile(prof, header):
    print()
    print(f"=== {header} ===")
    for name, t in prof.items():
        print(f"  {name:<20s} {t.summary()}")
    print()

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from stitcher.compositing import (
    composite_multiband_cpu,
    composite_multiband_gpu_async,
    get_pyr_kernel_2d,
)
from stitcher.device import detect_device
from stitcher.geometry import (
    build_remap,
    build_static_geometry,
    compute_canvas,
    estimate_homography,
    find_autocrop_rect,
)
from stitcher.io_utils import (
    PrefetchingFrameReader,
    ThreadedVideoWriter,
    draw_mask_overlay,
    draw_seam_overlay,
)
from stitcher.motion import (
    compute_mean_in_overlap_cpu,
    compute_mean_in_overlap_gpu,
    compute_motion_mask_cpu,
    compute_motion_mask_cpu_chrominance,
    compute_motion_mask_cpu_edges,
    compute_motion_mask_gpu,
    compute_motion_mask_gpu_chrominance,
    compute_motion_mask_gpu_edges,
    crop_to_bbox_cpu,
    crop_to_bbox_gpu,
    downsample_image_half_cpu,
    downsample_image_half_gpu,
    downsample_mask_half_cpu,
    downsample_mask_half_gpu,
    grab_baseline_from_videos,
    load_baseline_images,
    precompute_baseline_ab_cpu,
    precompute_baseline_ab_gpu,
    renormalize_to_baseline_cpu,
    renormalize_to_baseline_gpu,
    sobel_magnitude_cpu_bb,
    sobel_magnitude_gpu_bb,
    upsample_mask_to_bbox_cpu,
    upsample_mask_to_bbox_gpu,
    validate_baseline_shape,
)
from stitcher.seam import (
    add_edge_margin_penalty,
    add_seam_regularizer,
    compute_cost_and_ema_gpu,
    compute_cost_fast_cpu,
    find_dp_seam,
    upscale_seam,
)
from stitcher.segmentation import (
    PERSON_CLASS_ID,
    PersonSegmenter,
    compute_fg_mask_seg_cpu,
    compute_fg_mask_seg_gpu,
)
from stitcher.sync_reader import FrameSyncReader
from stitcher.warp import (
    apply_gain_lut,
    build_gain_lut,
    build_gain_tensor,
    build_grid_sample_tensor,
    compute_gain_compensation,
    dilate_gpu,
    warp_mask_gpu,
    warp_pair_gpu,
)


HOMOGRAPHY_PATH = "homography.npy"


def _resolve_relpath(p):
    """
    Resolve a relative video path by walking up from this file's directory
    until the path exists, so the script can be invoked from a sibling
    directory and still find project-local `videos/...` paths.
    """
    if os.path.isabs(p):
        return p
    if os.path.exists(p):
        return p
    cur = os.path.dirname(__file__)
    for _ in range(6):
        candidate = os.path.join(cur, p)
        if os.path.exists(candidate):
            return candidate
        cur = os.path.dirname(cur)
    return p


def _build_segmenters(args, dev):
    """
    Build one segmenter per task (person, FG). When both tasks pick the
    same model type, share a single underlying model.

    Returns (person_segmenter, person_class_ids, fg_segmenter, fg_class_ids).
    """
    person_yoloe = args.person_model == "yoloe"
    fg_yoloe = args.fg_model == "yoloe"

    def _mk_yolov8():
        return PersonSegmenter(args.yolo_weights, device=dev["yolo_device"])

    def _mk_yoloe(text_classes):
        return PersonSegmenter(
            args.yoloe_weights, device=dev["yolo_device"],
            use_yoloe=True, text_classes=text_classes,
        )

    if person_yoloe and fg_yoloe:
        text_classes = [args.yoloe_person_class] + list(args.yoloe_fg_classes)
        print(f"[info] Loading YOLOE for person + FG: {args.yoloe_weights}")
        print(f"[info] YOLOE classes (index 0 = person, 1+ = FG): {text_classes}")
        person_segmenter = _mk_yoloe(text_classes)
        fg_segmenter = person_segmenter
        person_class_ids = [0]
        fg_class_ids = list(range(1, len(text_classes)))
    elif not person_yoloe and not fg_yoloe:
        print(f"[info] Loading YOLOv8 for person + FG: {args.yolo_weights}")
        person_segmenter = _mk_yolov8()
        fg_segmenter = person_segmenter
        person_class_ids = [PERSON_CLASS_ID]
        fg_class_ids = list(args.fg_classes)
    elif person_yoloe and not fg_yoloe:
        print(f"[info] Loading YOLOE for person: {args.yoloe_weights} "
              f"(class: {args.yoloe_person_class!r})")
        person_segmenter = _mk_yoloe([args.yoloe_person_class])
        person_class_ids = [0]
        print(f"[info] Loading YOLOv8 for FG: {args.yolo_weights}")
        fg_segmenter = _mk_yolov8()
        fg_class_ids = list(args.fg_classes)
    else:  # not person_yoloe and fg_yoloe
        print(f"[info] Loading YOLOv8 for person: {args.yolo_weights}")
        person_segmenter = _mk_yolov8()
        person_class_ids = [PERSON_CLASS_ID]
        print(f"[info] Loading YOLOE for FG: {args.yoloe_weights} "
              f"(classes: {list(args.yoloe_fg_classes)})")
        fg_segmenter = _mk_yoloe(list(args.yoloe_fg_classes))
        fg_class_ids = list(range(len(args.yoloe_fg_classes)))

    return person_segmenter, person_class_ids, fg_segmenter, fg_class_ids


def _print_device_banner(dev):
    print("=" * 60)
    print("[device] Device detection")
    print("=" * 60)
    if dev["cuda_available"]:
        print(f"[device] CUDA: AVAILABLE")
        print(f"[device] GPU: {dev['gpu_name']}  ({dev['gpu_mem_gb']:.1f} GB)")
        print(f"[device] YOLO          -> GPU")
        print(f"[device] gain          -> GPU (fused with warp upload)")
        print(f"[device] warp          -> GPU (PyTorch grid_sample)")
        print(f"[device] mask warp+dil -> GPU (grid_sample + max_pool2d)")
        print(f"[device] cost+ema      -> GPU (PyTorch)")
        print(f"[device] composite     -> GPU (PyTorch, resident)")
        print(f"[device] DP seam       -> CPU (tiny)")
        print(f"[device] decode        -> CPU")
        print(f"[device] write         -> threaded CPU")
    else:
        print(f"[device] CUDA: NOT AVAILABLE — running entirely on CPU")
    print("=" * 60)


def run(args):
    """
    Run the full stitching pipeline using the parsed args. Argparse and
    flag definitions live in the entry script; this function only
    consumes the resulting Namespace.
    """
    dev = detect_device()
    _print_device_banner(dev)

    if not dev["cuda_available"]:
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

    print(f"[info] OpenCV: {cv2.getNumberOfCPUs()} CPUs, "
          f"using {cv2.getNumThreads()} threads.")
    print(f"[info] torch num_threads = {torch.get_num_threads()}")

    ema_eff = 1.0 if args.no_cost_ema else float(args.cost_ema)
    print(f"[info] yolo_every={args.yolo_every}  "
          f"mask_dilate={args.mask_dilate}  "
          f"DP_downscale={args.seam_downscale}  "
          f"gain_comp={not args.no_gain_comp}  "
          f"cost_ema={ema_eff}  "
          f"blend_width={args.blend_width}  "
          f"blend_levels={args.blend_levels}")
    print(f"[info] seam_lambda={args.seam_lambda}  "
          f"seam_edge_margin={args.seam_edge_margin}")

    args.video_a = _resolve_relpath(args.video_a)
    args.video_b = _resolve_relpath(args.video_b)

    cap_a = cv2.VideoCapture(args.video_a)
    cap_b = cv2.VideoCapture(args.video_b)
    if not cap_a.isOpened() or not cap_b.isOpened():
        raise RuntimeError("Could not open one of the input videos.")
    fps_a = cap_a.get(cv2.CAP_PROP_FPS) or 25.0
    fps_b = cap_b.get(cv2.CAP_PROP_FPS) or 25.0

    sync_reader = FrameSyncReader(cap_a, cap_b, fps_a, fps_b)
    print(sync_reader.summary())

    # Read the first paired frame (used for homography + gain seed).
    ok, frame_a, frame_b = sync_reader.read()
    if not ok:
        raise RuntimeError("Could not read first frame pair.")

    print("[info] Estimating homography from first frame pair...")
    H_b_to_a = estimate_homography(frame_a, frame_b)
    np.save(HOMOGRAPHY_PATH, H_b_to_a)

    canvas_size, T, H_b_to_canvas, H_a_to_canvas = compute_canvas(
        frame_a.shape, frame_b.shape, H_b_to_a
    )
    print(f"[info] Canvas size: {canvas_size[0]} x {canvas_size[1]}")

    # When --autocrop is on, push the crop translation through the
    # homographies. Everything downstream (remap maps, static masks,
    # overlap bbox, warp grids, warped frames, composite output, pinned
    # buffers) is then built at the cropped size — the warp itself
    # produces only the pixels that ship to disk, so the pixels outside
    # the crop never get computed at all. `output_size` replaces
    # `canvas_size` for every per-frame allocation.
    crop_rect = None
    if args.autocrop:
        crop_rect = find_autocrop_rect(
            H_b_to_a, frame_a.shape, frame_b.shape, canvas_size, T,
        )
        cx, cy, cw, ch = crop_rect
        print(f"[info] Autocrop: x={cx} y={cy} size={cw}x{ch} "
              f"(from full canvas {canvas_size[0]}x{canvas_size[1]})")
        T_crop = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]],
                          dtype=np.float64)
        H_a_to_canvas = T_crop @ H_a_to_canvas
        H_b_to_canvas = T_crop @ H_b_to_canvas
        output_size = (cw, ch)
    else:
        output_size = canvas_size

    print("[info] Precomputing remap maps + static geometry...")
    map_ax, map_ay = build_remap(H_a_to_canvas, output_size)
    map_bx, map_by = build_remap(H_b_to_canvas, output_size)
    static = build_static_geometry(
        frame_a.shape, frame_b.shape,
        map_ax, map_ay, map_bx, map_by,
        output_size,
    )
    x0, y0, x1, y1 = static["overlap_bbox"]
    bbox_shape = (y1 - y0, x1 - x0)
    print(f"[info] Overlap bbox: x=[{x0},{x1}) y=[{y0},{y1}) "
          f"size={bbox_shape[1]}x{bbox_shape[0]}")

    lut_a = None
    lut_b = None
    gain_a_t = None
    gain_b_t = None
    if not args.no_gain_comp:
        print("[info] Computing gain compensation from first frame pair...")
        wa0 = cv2.remap(frame_a, map_ax, map_ay, cv2.INTER_LINEAR)
        wb0 = cv2.remap(frame_b, map_bx, map_by, cv2.INTER_LINEAR)
        gains_a, gains_b = compute_gain_compensation(
            wa0, wb0, static["overlap_bbox"], static["overlap_in_bbox"]
        )
        print(f"[info] gains_a = [{gains_a[0]:.3f}, {gains_a[1]:.3f}, {gains_a[2]:.3f}]")
        print(f"[info] gains_b = [{gains_b[0]:.3f}, {gains_b[1]:.3f}, {gains_b[2]:.3f}]")
        if dev["cuda_available"]:
            gain_a_t = build_gain_tensor(gains_a, torch.device("cuda"))
            gain_b_t = build_gain_tensor(gains_b, torch.device("cuda"))
        else:
            lut_a = build_gain_lut(gains_a)
            lut_b = build_gain_lut(gains_b)

    person_segmenter, person_class_ids, fg_segmenter, fg_class_ids = (
        _build_segmenters(args, dev)
    )
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * args.mask_dilate + 1, 2 * args.mask_dilate + 1),
    )
    fg_dilate_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * args.fg_dilate + 1, 2 * args.fg_dilate + 1),
        ) if args.fg_dilate > 0 else None
    )

    gpu_ctx = None
    grid_a_t = None
    grid_b_t = None
    overlap_in_bbox_t = None
    cost_ema_t = None
    if dev["cuda_available"]:
        torch_device = torch.device("cuda")
        valid_in_bbox_np = cv2.bitwise_or(static["mask_a_in_bbox"],
                                          static["mask_b_in_bbox"])
        out_W, out_H = output_size
        gpu_ctx = {
            "device": torch_device,
            "kernel2d": get_pyr_kernel_2d(torch_device),
            "only_a_u8_t": torch.from_numpy(static["only_a_u8"]).to(torch_device),
            "only_b_u8_t": torch.from_numpy(static["only_b_u8"]).to(torch_device),
            "only_a_in_bbox_t": torch.from_numpy(static["only_a_in_bbox"]).to(torch_device),
            "only_b_in_bbox_t": torch.from_numpy(static["only_b_in_bbox"]).to(torch_device),
            "overlap_in_bbox_t": torch.from_numpy(static["overlap_in_bbox"]).to(torch_device),
            "valid_in_bbox_t": torch.from_numpy(valid_in_bbox_np).to(torch_device),
        }
        # Ring of pinned host buffers for the async composite path. Each
        # buffer holds one in-flight (out_H, out_W, 3) uint8 frame between
        # composite_one (writer of the pinned copy) and the writer
        # thread (encoder). Sized to cover composite_in_q (4) + writer
        # queue (4) + one in flight in each worker, with headroom.
        pinned_H, pinned_W = out_H, out_W
        pinned_ring_size = 10
        free_pinned_q = queue.Queue(maxsize=pinned_ring_size)
        pinned_ring = []  # keep references so they aren't GC'd
        for _ in range(pinned_ring_size):
            buf = torch.empty(
                (pinned_H, pinned_W, 3),
                dtype=torch.uint8, pin_memory=True,
            )
            pinned_ring.append(buf)
            free_pinned_q.put(buf)
        grid_a_t = build_grid_sample_tensor(map_ax, map_ay, frame_a.shape, torch_device)
        grid_b_t = build_grid_sample_tensor(map_bx, map_by, frame_b.shape, torch_device)
        # Stacked grid for the batched warp_pair_gpu call. Built once
        # since the grids are static for the whole run.
        grid_pair_t = torch.cat([grid_a_t, grid_b_t], dim=0)
        overlap_in_bbox_t = torch.from_numpy(static["overlap_in_bbox"]).to(torch_device)
        # Tier-2 pipeline parallelism: each worker thread runs its CUDA
        # work on its own stream so the GPU can interleave kernels from
        # consecutive frames (composite N + compute N+1) instead of
        # serialising them on the default stream.
        # Stream priorities: in PyTorch CUDA, priority -1 is HIGH and 0
        # is the (low) default. We mark the three critical-path streams
        # (compute, composite, yolo) as high and leave motion at the
        # default. That way when motion + YOLO have kernels queued on
        # the GPU at the same time, the scheduler runs YOLO first
        # instead of interleaving them at kernel granularity (which was
        # tripling YOLO's effective inference time when motion fires
        # every frame).
        compute_stream = torch.cuda.Stream(priority=-1)
        composite_stream = torch.cuda.Stream(priority=-1)
        yolo_stream = torch.cuda.Stream(priority=-1)
        # Motion worker is the only "background" stream — it can yield
        # to anything else that has queued work, since the motion mask
        # only needs to be ready a frame later for the next cost
        # computation, not immediately.
        motion_stream = torch.cuda.Stream(priority=0)
        print("[device] GPU contexts (gain + warp + mask + cost + composite) initialized.")

    # --- Static foreground mask (segmentation-based) -----------------------
    use_fg = not args.no_fg
    fg_mask_bbox_t = None  # GPU path
    fg_mask_bbox = None    # CPU path
    if use_fg:
        print(f"[info] Computing static FG mask via YOLO segmentation. "
              f"Class IDs: {fg_class_ids}")
        t0 = time.time()
        if dev["cuda_available"]:
            fg_mask_bbox_t = compute_fg_mask_seg_gpu(
                fg_segmenter, frame_a, frame_b, fg_class_ids,
                grid_a_t, grid_b_t, args.fg_dilate,
                static["overlap_bbox"], overlap_in_bbox_t,
            )
            coverage = (fg_mask_bbox_t > 0).float().mean().item() * 100
        else:
            fg_mask_bbox = compute_fg_mask_seg_cpu(
                fg_segmenter, frame_a, frame_b, fg_class_ids,
                map_ax, map_ay, map_bx, map_by,
                fg_dilate_kernel,
                static["overlap_bbox"], static["overlap_in_bbox"],
            )
            coverage = float((fg_mask_bbox > 0).mean()) * 100
        print(f"[info] FG mask computed in {(time.time()-t0)*1000:.1f} ms  "
              f"({coverage:.1f}% of bbox flagged).")

    fg_recompute_frames = (
        int(round(args.fg_recompute_seconds * sync_reader.output_fps))
        if args.fg_recompute_seconds > 0 else 0
    )
    if use_fg and fg_recompute_frames > 0:
        print(f"[info] FG recompute every {fg_recompute_frames} frames "
              f"(~{args.fg_recompute_seconds}s).")

    # --- Motion detection (baseline subtraction) --------------------------
    # --motion and --motion_renorm are now ON by default; the CLI flags
    # are inverted as --no_motion / --no_motion_renorm. Map back here so
    # the rest of the code keeps reading the positive predicate.
    args.motion = not bool(getattr(args, "no_motion", False))
    args.motion_renorm = not bool(getattr(args, "no_motion_renorm", False))
    use_motion = bool(args.motion)
    motion_dilate_kernel = None
    # All motion-side baselines are stored already cropped to the
    # overlap bbox — the per-frame mask helpers operate on bbox-cropped
    # tensors (see stitcher.motion), so cropping once at startup
    # eliminates a slice per frame on the hot path.
    baseline_a_bb_t = None       # GPU pixel-method baseline (3, H_bb, W_bb)
    baseline_b_bb_t = None
    baseline_a_bb = None         # CPU pixel-method baseline (H_bb, W_bb, 3)
    baseline_b_bb = None
    baseline_grad_a_bb_t = None  # GPU edge-method baseline (H_bb, W_bb) float
    baseline_grad_b_bb_t = None
    baseline_grad_a_bb = None    # CPU edge-method baseline (H_bb, W_bb) float
    baseline_grad_b_bb = None
    baseline_mean_a_t = None     # GPU renorm baseline mean BGR (3,)
    baseline_mean_b_t = None
    baseline_mean_a = None       # CPU renorm baseline mean BGR (3,)
    baseline_mean_b = None
    baseline_ab_a_bb_t = None    # GPU chrominance baseline (2, H_bb, W_bb)
    baseline_ab_b_bb_t = None
    baseline_ab_a_bb = None      # CPU chrominance baseline (H_bb, W_bb, 2)
    baseline_ab_b_bb = None
    if use_motion:
        paths_a = args.motion_baseline_a
        paths_b = args.motion_baseline_b
        if (paths_a is None) != (paths_b is None):
            raise RuntimeError(
                "--motion_baseline_a and --motion_baseline_b must be "
                "provided together (or both omitted, to fall back to "
                "frame 0 of each video)."
            )
        if paths_a is not None:
            print(f"[info] Loading motion baselines: A={paths_a} B={paths_b}")
            baseline_frame_a, baseline_frame_b = load_baseline_images(
                paths_a, paths_b,
            )
        else:
            print("[info] Motion baselines not provided; falling back to "
                  "frame 0 of each video.")
            baseline_frame_a, baseline_frame_b = grab_baseline_from_videos(
                args.video_a, args.video_b,
            )
        validate_baseline_shape(
            frame_a, frame_b, baseline_frame_a, baseline_frame_b,
        )

        motion_dilate_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * args.motion_dilate + 1, 2 * args.motion_dilate + 1),
            ) if args.motion_dilate > 0 else None
        )

        if dev["cuda_available"]:
            # Warp both baselines through grid_pair_t, crop to the
            # overlap bbox, then downsample bbox -> half-res. The
            # per-frame motion diff runs at half-res to cut work ~4x;
            # see motion_worker for the rest of the half-res path.
            full_baseline_a_t, full_baseline_b_t = warp_pair_gpu(
                baseline_frame_a, baseline_frame_b,
                grid_pair_t, gpu_ctx["device"],
                gain_a_t=gain_a_t, gain_b_t=gain_b_t,
            )
            baseline_a_bb_t = crop_to_bbox_gpu(full_baseline_a_t,
                                               static["overlap_bbox"])
            baseline_b_bb_t = crop_to_bbox_gpu(full_baseline_b_t,
                                               static["overlap_bbox"])
            # Half-res baselines + overlap mask (all per-frame work
            # operates on these).
            baseline_a_bb_t = downsample_image_half_gpu(baseline_a_bb_t)
            baseline_b_bb_t = downsample_image_half_gpu(baseline_b_bb_t)
            overlap_in_bbox_motion_t = downsample_mask_half_gpu(overlap_in_bbox_t)
            if args.motion_method == "edges":
                baseline_grad_a_bb_t = sobel_magnitude_gpu_bb(baseline_a_bb_t)
                baseline_grad_b_bb_t = sobel_magnitude_gpu_bb(baseline_b_bb_t)
            elif args.motion_method == "chrominance":
                baseline_ab_a_bb_t = precompute_baseline_ab_gpu(baseline_a_bb_t)
                baseline_ab_b_bb_t = precompute_baseline_ab_gpu(baseline_b_bb_t)
            if args.motion_renorm:
                baseline_mean_a_t = compute_mean_in_overlap_gpu(
                    baseline_a_bb_t, overlap_in_bbox_motion_t,
                )
                baseline_mean_b_t = compute_mean_in_overlap_gpu(
                    baseline_b_bb_t, overlap_in_bbox_motion_t,
                )
            del full_baseline_a_t, full_baseline_b_t
        else:
            if lut_a is not None:
                baseline_frame_a = apply_gain_lut(baseline_frame_a, lut_a)
                baseline_frame_b = apply_gain_lut(baseline_frame_b, lut_b)
            full_baseline_a = cv2.remap(baseline_frame_a, map_ax, map_ay,
                                        cv2.INTER_LINEAR)
            full_baseline_b = cv2.remap(baseline_frame_b, map_bx, map_by,
                                        cv2.INTER_LINEAR)
            baseline_a_bb = crop_to_bbox_cpu(full_baseline_a,
                                             static["overlap_bbox"])
            baseline_b_bb = crop_to_bbox_cpu(full_baseline_b,
                                             static["overlap_bbox"])
            # Half-res baselines + overlap mask for the CPU motion path.
            baseline_a_bb = downsample_image_half_cpu(baseline_a_bb)
            baseline_b_bb = downsample_image_half_cpu(baseline_b_bb)
            overlap_in_bbox_motion = downsample_mask_half_cpu(
                static["overlap_in_bbox"]
            )
            if args.motion_method == "edges":
                baseline_grad_a_bb = sobel_magnitude_cpu_bb(baseline_a_bb)
                baseline_grad_b_bb = sobel_magnitude_cpu_bb(baseline_b_bb)
            elif args.motion_method == "chrominance":
                baseline_ab_a_bb = precompute_baseline_ab_cpu(baseline_a_bb)
                baseline_ab_b_bb = precompute_baseline_ab_cpu(baseline_b_bb)
            if args.motion_renorm:
                baseline_mean_a = compute_mean_in_overlap_cpu(
                    baseline_a_bb, overlap_in_bbox_motion,
                )
                baseline_mean_b = compute_mean_in_overlap_cpu(
                    baseline_b_bb, overlap_in_bbox_motion,
                )
        print(f"[info] Motion: method={args.motion_method} "
              f"renorm={args.motion_renorm} "
              f"threshold={args.motion_threshold} "
              f"dilate={args.motion_dilate} penalty={args.motion_penalty:g}  "
              f"(running at 1/{2}-res inside the bbox)")

    W, H = output_size
    out_buf      = np.zeros((H, W, 3), dtype=np.uint8)
    cost_scratch = np.empty((bbox_shape[0], bbox_shape[1], 3), dtype=np.float32)
    # Person mask + EMA state now lives in the yolo_worker thread (see
    # below). compute_one reads the latest published mask from
    # mask_holder via mask_lock.
    cost_ema = None
    seam_prev_small = None

    # Async YOLO handoff: the yolo_worker publishes mask snapshots into
    # mask_holder under mask_lock; compute_one grabs the latest available.
    # On the first few frames before the worker has run, the holder is
    # None and compute_one proceeds with no person penalty (the seam is
    # briefly unaware of people — acceptable for a few frames).
    mask_lock = threading.Lock()
    mask_holder = [None]
    yolo_q = queue.Queue(maxsize=1)

    # Async motion handoff: the motion_worker (only spawned when
    # --motion is set) takes warped tensors + an event recorded at end
    # of warp on compute_stream, computes the bbox motion mask on its
    # own stream, and publishes the result here. compute_one reads the
    # latest published mask; it is from the PREVIOUS frame (1-frame
    # lag), which at 25+ fps is invisible compared to dilate_radius.
    motion_mask_lock = threading.Lock()
    motion_mask_holder = [None]
    motion_q = queue.Queue(maxsize=1)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    raw_writer = cv2.VideoWriter(args.output, fourcc, sync_reader.output_fps, output_size)
    if not raw_writer.isOpened():
        raise RuntimeError(f"Could not open output writer for {args.output}")
    writer = ThreadedVideoWriter(raw_writer, queue_depth=4)

    # Reuse the homography frame as the first iteration; subsequent
    # iterations pull from a background thread that decodes the next
    # paired frame in parallel with the compute loop.
    pending_first_pair = (frame_a, frame_b)
    prefetch_reader = PrefetchingFrameReader(sync_reader, queue_depth=3)

    # --- Per-frame compute and composite, as nested closures ---------------
    #
    # The pipeline runs two worker threads (compute_worker and
    # composite_worker) plus the prefetch decode thread (PrefetchingFrameReader)
    # and the write thread (ThreadedVideoWriter). Per-frame work is split at
    # the natural data boundary: everything that produces the seam and the
    # warped frames (compute_one) vs everything that consumes them to make
    # the final stitched image (composite_one). The two stages run on
    # consecutive frames at the same time.

    def compute_one(frame_a, frame_b, frame_idx):
        """Warp + cost + EMA + DP seam. YOLO runs async on its own
        worker; we submit a request when it's time and read the latest
        published person mask from mask_holder.

        Updates inter-frame state in the enclosing scope. Returns a
        payload dict consumed by composite_one."""
        nonlocal cost_ema_t, cost_ema
        nonlocal seam_prev_small
        nonlocal fg_mask_bbox_t, fg_mask_bbox

        # Periodic FG recompute.
        if (use_fg and fg_recompute_frames > 0 and frame_idx > 0
                and frame_idx % fg_recompute_frames == 0):
            if dev["cuda_available"]:
                fg_mask_bbox_t = compute_fg_mask_seg_gpu(
                    fg_segmenter, frame_a, frame_b, fg_class_ids,
                    grid_a_t, grid_b_t, args.fg_dilate,
                    static["overlap_bbox"], overlap_in_bbox_t,
                )
            else:
                fg_mask_bbox = compute_fg_mask_seg_cpu(
                    fg_segmenter, frame_a, frame_b, fg_class_ids,
                    map_ax, map_ay, map_bx, map_by,
                    fg_dilate_kernel,
                    static["overlap_bbox"], static["overlap_in_bbox"],
                )

        # Warp.
        warped_a_t = warped_b_t = None
        warped_a = warped_b = None
        if dev["cuda_available"]:
            warped_a_t, warped_b_t = warp_pair_gpu(
                frame_a, frame_b, grid_pair_t, gpu_ctx["device"],
                gain_a_t=gain_a_t, gain_b_t=gain_b_t,
            )
        else:
            if lut_a is not None:
                frame_a_g = apply_gain_lut(frame_a, lut_a)
                frame_b_g = apply_gain_lut(frame_b, lut_b)
            else:
                frame_a_g = frame_a
                frame_b_g = frame_b
            warped_a = cv2.remap(frame_a_g, map_ax, map_ay, cv2.INTER_LINEAR)
            warped_b = cv2.remap(frame_b_g, map_bx, map_by, cv2.INTER_LINEAR)

        # Async motion: submit this frame's warped tensors to the motion
        # worker (best-effort — if its queue is full, the worker is
        # still busy with the previous request and we just keep using
        # the last published mask). For GPU we also hand it an event
        # recorded right after warp so the worker's stream can wait on
        # it without a host stall.
        if use_motion:
            try:
                if dev["cuda_available"]:
                    warp_done_event = torch.cuda.Event()
                    warp_done_event.record()
                    motion_q.put_nowait(
                        (warped_a_t, warped_b_t, warp_done_event)
                    )
                else:
                    motion_q.put_nowait((warped_a, warped_b))
            except queue.Full:
                pass

        # Async YOLO: submit a request when it's time, then continue with
        # whichever mask is currently published. The first few frames
        # before YOLO has produced anything see latest_mask = None and
        # run without a person penalty.
        if frame_idx % args.yolo_every == 0:
            try:
                yolo_q.put_nowait((frame_a, frame_b))
            except queue.Full:
                # YOLO worker is still processing the previous request;
                # skip this submission and reuse the latest mask we have.
                pass

        with mask_lock:
            latest_mask = mask_holder[0]

        person_mask_bbox_t = None
        person_mask_bbox = None
        person_for_debug = None
        if latest_mask is not None:
            if dev["cuda_available"]:
                ev = latest_mask.get("ready_event")
                if ev is not None:
                    ev.wait()
                person_mask_bbox_t = latest_mask["person_mask_bbox_t"]
            else:
                person_mask_bbox = latest_mask["person_mask_bbox"]
            if args.debug_mask:
                person_for_debug = latest_mask.get("person_mask_cpu")

        # Motion mask comes from the async motion_worker. Read the most
        # recent published mask (it's from frame N-1 by construction);
        # for the first frame or two before the worker has anything to
        # publish, the holder is None and the seam runs without a
        # motion penalty — same warm-up pattern as the YOLO mask.
        motion_mask_bbox_t = None
        motion_mask_bbox = None
        motion_for_debug = None
        if use_motion:
            with motion_mask_lock:
                latest_motion = motion_mask_holder[0]
            if latest_motion is not None:
                if dev["cuda_available"]:
                    ev = latest_motion.get("ready_event")
                    if ev is not None:
                        ev.wait()
                    motion_mask_bbox_t = latest_motion["motion_mask_bbox_t"]
                else:
                    motion_mask_bbox = latest_motion["motion_mask_bbox"]
                if args.debug_mask:
                    motion_for_debug = latest_motion.get("motion_mask_cpu")

        ds = max(1, args.seam_downscale)

        if dev["cuda_available"]:
            has_person = (person_mask_bbox_t.any().item()
                          if person_mask_bbox_t is not None else False)
            cost_ema_t, cost_for_dp_t = compute_cost_and_ema_gpu(
                warped_a_t, warped_b_t, overlap_in_bbox_t,
                cost_ema_t, ema_eff,
                person_mask_bbox_t if has_person else None,
                fg_mask_bbox_t if use_fg else None,
                args.fg_penalty, args.person_penalty,
                static["overlap_bbox"],
                motion_mask_bbox_t=motion_mask_bbox_t,
                motion_penalty=args.motion_penalty,
            )
            if args.seam_edge_margin > 0:
                m = min(args.seam_edge_margin,
                        cost_for_dp_t.shape[1] // 2)
                cost_for_dp_t[:, :m] += args.edge_penalty
                cost_for_dp_t[:, -m:] += args.edge_penalty
            if ds > 1:
                cost_small_t = F.avg_pool2d(
                    cost_for_dp_t.unsqueeze(0).unsqueeze(0),
                    kernel_size=ds, stride=ds,
                )[0, 0]
            else:
                cost_small_t = cost_for_dp_t
            cost_small = cost_small_t.cpu().numpy()
        else:
            wa_bb = warped_a[y0:y1, x0:x1]
            wb_bb = warped_b[y0:y1, x0:x1]
            photo_cost = compute_cost_fast_cpu(
                wa_bb, wb_bb, static["overlap_in_bbox"], cost_scratch,
            )
            if ema_eff >= 1.0 or cost_ema is None:
                if cost_ema is None or cost_ema.shape != photo_cost.shape:
                    cost_ema = photo_cost.copy()
                else:
                    np.copyto(cost_ema, photo_cost)
            else:
                cv2.addWeighted(photo_cost, ema_eff,
                                cost_ema, 1.0 - ema_eff,
                                0, dst=cost_ema)
            cost_for_dp = cost_ema.copy()
            has_person_cpu = (person_mask_bbox is not None
                              and person_mask_bbox.any())
            # Mirror the GPU penalty hierarchy: fg (lower) and motion
            # (lower) where mask AND NOT person, then person_penalty.
            if use_fg and fg_mask_bbox is not None:
                fg_bool = fg_mask_bbox > 0
                fg_only = (fg_bool & (person_mask_bbox == 0)
                           if has_person_cpu else fg_bool)
                cost_for_dp[fg_only] += args.fg_penalty
            if use_motion and motion_mask_bbox is not None:
                motion_bool = motion_mask_bbox > 0
                motion_only = (motion_bool & (person_mask_bbox == 0)
                               if has_person_cpu else motion_bool)
                cost_for_dp[motion_only] += args.motion_penalty
            if has_person_cpu:
                cost_for_dp[person_mask_bbox > 0] += args.person_penalty
            add_edge_margin_penalty(cost_for_dp, args.seam_edge_margin,
                                    edge_penalty=args.edge_penalty)
            if ds > 1:
                cost_small = cv2.resize(
                    cost_for_dp,
                    (cost_for_dp.shape[1] // ds, cost_for_dp.shape[0] // ds),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                cost_small = cost_for_dp.copy()

        add_seam_regularizer(cost_small, seam_prev_small, args.seam_lambda)
        seam_x_small = find_dp_seam(cost_small)
        seam_prev_small = seam_x_small.copy()
        seam_x_full = upscale_seam(seam_x_small, bbox_shape, ds)

        # Snapshot the FG mask for the debug overlay (composite stage
        # may run a frame later, so we capture the version that's
        # current for THIS frame here). person_for_debug and
        # motion_for_debug were already set above when we read their
        # respective holders.
        fg_for_debug = None
        if args.debug_mask and use_fg:
            if fg_mask_bbox_t is not None:
                fg_for_debug = fg_mask_bbox_t.cpu().numpy()
            elif fg_mask_bbox is not None:
                fg_for_debug = fg_mask_bbox

        # Record an event on the current (compute) stream so the composite
        # stage's stream can wait on it before reading our output tensors.
        compute_done_event = None
        if dev["cuda_available"]:
            compute_done_event = torch.cuda.Event()
            compute_done_event.record()

        return {
            "frame_idx": frame_idx,
            "warped_a_t": warped_a_t,
            "warped_b_t": warped_b_t,
            "warped_a": warped_a,
            "warped_b": warped_b,
            "seam_x_full": seam_x_full,
            "person_for_debug": person_for_debug,
            "fg_for_debug": fg_for_debug,
            "motion_for_debug": motion_for_debug,
            "compute_done_event": compute_done_event,
        }

    def composite_one(payload):
        """Run the multi-band composite + debug overlays + autocrop.

        On GPU: returns ("async", pinned, event, post_sync_fn, free_cb)
        — composite_worker hands this off to writer.write_async, which
        synchronises on the event before encoding. composite_one itself
        does NOT wait for the GPU.

        On CPU: returns ("sync", stitched_ndarray) — the writer takes
        the already-materialized frame.
        """
        # Wait for the compute stage's output tensors to be ready (their
        # writes happened on compute_stream; we read them on
        # composite_stream). Stream wait is the cheap, GPU-side
        # synchronisation primitive — no host stall.
        compute_event = payload.get("compute_done_event")
        if compute_event is not None:
            compute_event.wait()
        seam_x_full = payload["seam_x_full"]
        if gpu_ctx is not None:
            # Acquire a pinned slot (blocks if all are in flight — this
            # is our backpressure mechanism between composite and writer).
            pinned = free_pinned_q.get()
            copy_event = composite_multiband_gpu_async(
                payload["warped_a_t"], payload["warped_b_t"],
                static, seam_x_full,
                args.blend_width, args.blend_levels,
                pinned, gpu_ctx,
            )
            fg_for_debug = payload["fg_for_debug"]
            person_for_debug = payload["person_for_debug"]
            motion_for_debug = payload["motion_for_debug"]

            # Closure runs on the writer thread AFTER copy_event has
            # been synchronised; sees the pinned numpy view as `arr`.
            # static["overlap_bbox"] is already in output (crop) coords
            # because the geometry was rebuilt at output_size up front.
            # Debug overlay layering: FG (yellow) < motion (blue) < person (red).
            def post_sync_fn(arr,
                             seam_x_full=seam_x_full,
                             fg_for_debug=fg_for_debug,
                             motion_for_debug=motion_for_debug,
                             person_for_debug=person_for_debug):
                if args.debug_mask:
                    if fg_for_debug is not None:
                        draw_mask_overlay(arr, fg_for_debug,
                                          static["overlap_bbox"],
                                          color=(0, 255, 255), alpha=0.25)
                    if motion_for_debug is not None:
                        draw_mask_overlay(arr, motion_for_debug,
                                          static["overlap_bbox"],
                                          color=(255, 0, 0), alpha=0.30)
                    if person_for_debug is not None:
                        draw_mask_overlay(arr, person_for_debug,
                                          static["overlap_bbox"])
                if args.debug_seam:
                    draw_seam_overlay(arr, seam_x_full,
                                      static["overlap_bbox"])
                return arr

            return ("async", pinned, copy_event, post_sync_fn,
                    lambda p=pinned: free_pinned_q.put(p))

        # ---- CPU path: unchanged synchronous behaviour. -----------------
        stitched = composite_multiband_cpu(
            payload["warped_a"], payload["warped_b"],
            static, seam_x_full,
            args.blend_width, args.blend_levels, out_buf,
        )
        if args.debug_mask:
            # Layer FG (yellow) < motion (blue) < person (red).
            fg_for_debug = payload["fg_for_debug"]
            if fg_for_debug is not None:
                draw_mask_overlay(stitched, fg_for_debug,
                                  static["overlap_bbox"],
                                  color=(0, 255, 255), alpha=0.25)
            motion_for_debug = payload["motion_for_debug"]
            if motion_for_debug is not None:
                draw_mask_overlay(stitched, motion_for_debug,
                                  static["overlap_bbox"],
                                  color=(255, 0, 0), alpha=0.30)
            person_for_debug = payload["person_for_debug"]
            if person_for_debug is not None:
                draw_mask_overlay(stitched, person_for_debug,
                                  static["overlap_bbox"])
        if args.debug_seam:
            draw_seam_overlay(stitched, seam_x_full, static["overlap_bbox"])
        return ("sync", stitched)

    # --- Pipelined execution with three worker threads -------------------
    SENTINEL = object()
    compute_in_q = queue.Queue(maxsize=4)
    composite_in_q = queue.Queue(maxsize=4)
    worker_error = [None]

    # Optional per-stage timing. None when --profile is off (zero
    # overhead: each timing site checks `if prof is not None`).
    prof = _make_profile() if args.profile else None
    prof_stop = threading.Event()

    def yolo_worker():
        """Pull (frame_a, frame_b) pairs off yolo_q, run batched YOLO
        inference + mask warp/dilate/EMA/binarize, and publish the
        resulting mask snapshot to mask_holder under mask_lock.

        Owns the inter-call person-mask EMA state. compute_one only
        ever reads the published mask — it never updates this state.
        Running here on its own CUDA stream so the mask post-processing
        kernels can overlap with the compute and composite streams."""
        person_mask_ema_t = None  # GPU path
        person_mask_ema = None    # CPU path

        stream_ctx = (
            torch.cuda.stream(yolo_stream)
            if dev["cuda_available"]
            else _nullcontext()
        )
        try:
            with stream_ctx:
                while True:
                    t_get0 = time.perf_counter()
                    item = yolo_q.get()
                    if prof is not None:
                        prof["yolo_get_wait"].record(
                            (time.perf_counter() - t_get0) * 1000
                        )
                    if item is SENTINEL:
                        return
                    frame_a_yolo, frame_b_yolo = item
                    t_work0 = time.perf_counter()

                    if dev["cuda_available"]:
                        mask_a_src_t, mask_b_src_t = (
                            person_segmenter.predict_classes_mask_pair_gpu(
                                frame_a_yolo, frame_b_yolo,
                                frame_a_yolo.shape[:2], frame_b_yolo.shape[:2],
                                person_class_ids,
                            )
                        )
                        mask_a_canvas_t = warp_mask_gpu(mask_a_src_t, grid_a_t)
                        mask_b_canvas_t = warp_mask_gpu(mask_b_src_t, grid_b_t)
                        union_t = torch.bitwise_or(mask_a_canvas_t, mask_b_canvas_t)
                        union_t = dilate_gpu(union_t, args.mask_dilate)
                        raw_mask_t = union_t[y0:y1, x0:x1].contiguous()
                        new_float = (raw_mask_t > 0).float()
                        if person_mask_ema_t is None:
                            person_mask_ema_t = new_float
                        else:
                            a = args.mask_ema
                            person_mask_ema_t = (a * new_float
                                                 + (1.0 - a) * person_mask_ema_t)
                        person_mask_bbox_t = (
                            (person_mask_ema_t > args.mask_ema_threshold)
                            .to(torch.uint8) * 255
                        ).contiguous()
                        person_mask_cpu = (
                            person_mask_bbox_t.cpu().numpy()
                            if args.debug_mask else None
                        )
                        # Event the compute thread waits on before reading
                        # person_mask_bbox_t (cross-stream sync, no host stall).
                        ready_event = torch.cuda.Event()
                        ready_event.record()
                        new_holder = {
                            "person_mask_bbox_t": person_mask_bbox_t,
                            "person_mask_bbox": None,
                            "person_mask_cpu": person_mask_cpu,
                            "ready_event": ready_event,
                        }
                    else:
                        mask_a_src, mask_b_src = (
                            person_segmenter.predict_classes_mask_pair(
                                frame_a_yolo, frame_b_yolo, person_class_ids,
                            )
                        )
                        mask_a_canvas = cv2.remap(mask_a_src, map_ax, map_ay,
                                                  cv2.INTER_NEAREST)
                        mask_b_canvas = cv2.remap(mask_b_src, map_bx, map_by,
                                                  cv2.INTER_NEAREST)
                        union = cv2.bitwise_or(mask_a_canvas, mask_b_canvas)
                        union = cv2.dilate(union, dilate_kernel)
                        raw_mask = union[y0:y1, x0:x1]
                        new_float = (raw_mask > 0).astype(np.float32)
                        if person_mask_ema is None:
                            person_mask_ema = new_float
                        else:
                            a = args.mask_ema
                            person_mask_ema = (a * new_float
                                               + (1.0 - a) * person_mask_ema)
                        person_mask_bbox = (
                            (person_mask_ema > args.mask_ema_threshold)
                            .astype(np.uint8) * 255
                        )
                        new_holder = {
                            "person_mask_bbox_t": None,
                            "person_mask_bbox": person_mask_bbox,
                            "person_mask_cpu": (
                                person_mask_bbox if args.debug_mask else None
                            ),
                            "ready_event": None,
                        }

                    with mask_lock:
                        mask_holder[0] = new_holder
                    if prof is not None:
                        prof["yolo"].record(
                            (time.perf_counter() - t_work0) * 1000
                        )
        except Exception as e:
            worker_error[0] = e

    def motion_worker():
        """Pull (warped_a_t, warped_b_t, warp_done_event) off motion_q,
        compute the bbox-cropped motion mask on motion_stream, and
        publish to motion_mask_holder.

        Running here on its own CUDA stream so the diff / threshold /
        dilate kernels can interleave with the compute and composite
        streams, and so the host time spent in compute_one for the
        motion path drops to a queue-put + a holder-read.

        The published mask is from the previous frame (compute_one
        submits frame N's warped tensors but reads what was published
        for frame N-1); at 25+ fps the one-frame lag is well below the
        motion_dilate radius that already absorbs sub-pixel jitter.
        """
        stream_ctx = (
            torch.cuda.stream(motion_stream)
            if dev["cuda_available"]
            else _nullcontext()
        )
        try:
            with stream_ctx:
                while True:
                    item = motion_q.get()
                    if item is SENTINEL:
                        return

                    # Dilate radius is halved on the half-res grid so the
                    # mask grows to the same effective px on the final
                    # bbox after nearest upsample (10px -> 5px @ half-res
                    # -> 10px footprint after 2x nearest upsample).
                    half_dilate_radius = max(1, args.motion_dilate // 2)
                    if dev["cuda_available"]:
                        wa_full_t, wb_full_t, warp_event = item
                        if warp_event is not None:
                            warp_event.wait()
                        wa_bb_t = crop_to_bbox_gpu(wa_full_t,
                                                   static["overlap_bbox"])
                        wb_bb_t = crop_to_bbox_gpu(wb_full_t,
                                                   static["overlap_bbox"])
                        # Drop to half-res for the diff + dilate.
                        wa_bb_t = downsample_image_half_gpu(wa_bb_t)
                        wb_bb_t = downsample_image_half_gpu(wb_bb_t)
                        if args.motion_renorm:
                            wa_bb_t = renormalize_to_baseline_gpu(
                                wa_bb_t, baseline_mean_a_t,
                                overlap_in_bbox_motion_t,
                            )
                            wb_bb_t = renormalize_to_baseline_gpu(
                                wb_bb_t, baseline_mean_b_t,
                                overlap_in_bbox_motion_t,
                            )
                        if args.motion_method == "edges":
                            motion_half_t = compute_motion_mask_gpu_edges(
                                wa_bb_t, wb_bb_t,
                                baseline_grad_a_bb_t, baseline_grad_b_bb_t,
                                args.motion_threshold, half_dilate_radius,
                                overlap_in_bbox_motion_t,
                            )
                        elif args.motion_method == "chrominance":
                            motion_half_t = compute_motion_mask_gpu_chrominance(
                                wa_bb_t, wb_bb_t,
                                baseline_ab_a_bb_t, baseline_ab_b_bb_t,
                                args.motion_threshold, half_dilate_radius,
                                overlap_in_bbox_motion_t,
                            )
                        else:  # "pixel"
                            motion_half_t = compute_motion_mask_gpu(
                                wa_bb_t, wb_bb_t,
                                baseline_a_bb_t, baseline_b_bb_t,
                                args.motion_threshold, half_dilate_radius,
                                overlap_in_bbox_motion_t,
                            )
                        # Back up to full bbox res via nearest upsample.
                        motion_bb_t = upsample_mask_to_bbox_gpu(
                            motion_half_t, overlap_in_bbox_t.shape,
                        )
                        motion_cpu = (motion_bb_t.cpu().numpy()
                                      if args.debug_mask else None)
                        ready_event = torch.cuda.Event()
                        ready_event.record()
                        new_holder = {
                            "motion_mask_bbox_t": motion_bb_t,
                            "motion_mask_bbox": None,
                            "motion_mask_cpu": motion_cpu,
                            "ready_event": ready_event,
                        }
                    else:
                        wa_full, wb_full = item
                        wa_bb = crop_to_bbox_cpu(wa_full,
                                                 static["overlap_bbox"])
                        wb_bb = crop_to_bbox_cpu(wb_full,
                                                 static["overlap_bbox"])
                        wa_bb = downsample_image_half_cpu(wa_bb)
                        wb_bb = downsample_image_half_cpu(wb_bb)
                        if args.motion_renorm:
                            wa_bb = renormalize_to_baseline_cpu(
                                wa_bb, baseline_mean_a,
                                overlap_in_bbox_motion,
                            )
                            wb_bb = renormalize_to_baseline_cpu(
                                wb_bb, baseline_mean_b,
                                overlap_in_bbox_motion,
                            )
                        # Build a half-res dilate kernel for the CPU path.
                        if args.motion_dilate > 0:
                            half_motion_dilate_kernel = cv2.getStructuringElement(
                                cv2.MORPH_ELLIPSE,
                                (2 * half_dilate_radius + 1,
                                 2 * half_dilate_radius + 1),
                            )
                        else:
                            half_motion_dilate_kernel = None
                        if args.motion_method == "edges":
                            motion_half = compute_motion_mask_cpu_edges(
                                wa_bb, wb_bb,
                                baseline_grad_a_bb, baseline_grad_b_bb,
                                args.motion_threshold, half_motion_dilate_kernel,
                                overlap_in_bbox_motion,
                            )
                        elif args.motion_method == "chrominance":
                            motion_half = compute_motion_mask_cpu_chrominance(
                                wa_bb, wb_bb,
                                baseline_ab_a_bb, baseline_ab_b_bb,
                                args.motion_threshold, half_motion_dilate_kernel,
                                overlap_in_bbox_motion,
                            )
                        else:  # "pixel"
                            motion_half = compute_motion_mask_cpu(
                                wa_bb, wb_bb,
                                baseline_a_bb, baseline_b_bb,
                                args.motion_threshold, half_motion_dilate_kernel,
                                overlap_in_bbox_motion,
                            )
                        motion_bb = upsample_mask_to_bbox_cpu(
                            motion_half, static["overlap_in_bbox"].shape,
                        )
                        new_holder = {
                            "motion_mask_bbox_t": None,
                            "motion_mask_bbox": motion_bb,
                            "motion_mask_cpu": (
                                motion_bb if args.debug_mask else None
                            ),
                            "ready_event": None,
                        }

                    with motion_mask_lock:
                        motion_mask_holder[0] = new_holder
        except Exception as e:
            worker_error[0] = e

    def compute_worker():
        # Pin this thread's CUDA stream so every CUDA op in compute_one
        # runs on compute_stream and can overlap with composite_stream.
        stream_ctx = (
            torch.cuda.stream(compute_stream)
            if dev["cuda_available"]
            else _nullcontext()
        )
        try:
            with stream_ctx:
                while True:
                    t_get0 = time.perf_counter()
                    item = compute_in_q.get()
                    if prof is not None:
                        prof["compute_get_wait"].record(
                            (time.perf_counter() - t_get0) * 1000
                        )
                    if item is SENTINEL:
                        composite_in_q.put(SENTINEL)
                        return
                    fa, fb, idx = item
                    t_work0 = time.perf_counter()
                    payload = compute_one(fa, fb, idx)
                    if prof is not None:
                        prof["compute"].record(
                            (time.perf_counter() - t_work0) * 1000
                        )
                    t_put0 = time.perf_counter()
                    composite_in_q.put(payload)
                    if prof is not None:
                        prof["compute_put_wait"].record(
                            (time.perf_counter() - t_put0) * 1000
                        )
        except Exception as e:
            worker_error[0] = e
            composite_in_q.put(SENTINEL)

    def composite_worker():
        stream_ctx = (
            torch.cuda.stream(composite_stream)
            if dev["cuda_available"]
            else _nullcontext()
        )
        try:
            with stream_ctx:
                while True:
                    t_get0 = time.perf_counter()
                    item = composite_in_q.get()
                    if prof is not None:
                        prof["composite_get_wait"].record(
                            (time.perf_counter() - t_get0) * 1000
                        )
                    if item is SENTINEL:
                        return
                    t_work0 = time.perf_counter()
                    result = composite_one(item)
                    if prof is not None:
                        prof["composite"].record(
                            (time.perf_counter() - t_work0) * 1000
                        )
                    t_write0 = time.perf_counter()
                    if result[0] == "async":
                        _, pinned, event, post_sync_fn, free_cb = result
                        writer.write_async(pinned, event,
                                           post_sync_fn, free_cb)
                    else:
                        _, stitched = result
                        writer.write(stitched)
                    if prof is not None:
                        prof["composite_write"].record(
                            (time.perf_counter() - t_write0) * 1000
                        )
        except Exception as e:
            worker_error[0] = e

    compute_thread = threading.Thread(
        target=compute_worker, name="compute", daemon=True,
    )
    composite_thread = threading.Thread(
        target=composite_worker, name="composite", daemon=True,
    )
    yolo_thread = threading.Thread(
        target=yolo_worker, name="yolo", daemon=True,
    )
    # Motion worker is only spawned when --motion is set; otherwise the
    # thread, its stream, and the queue are all idle so there's no
    # point paying for them.
    motion_thread = None
    if use_motion:
        motion_thread = threading.Thread(
            target=motion_worker, name="motion", daemon=True,
        )
    compute_thread.start()
    composite_thread.start()
    yolo_thread.start()
    if motion_thread is not None:
        motion_thread.start()

    def profile_printer():
        # Rolling print every args.profile_interval seconds until shutdown.
        while not prof_stop.wait(args.profile_interval):
            _print_profile(prof, "rolling profile")

    profile_thread = None
    if prof is not None:
        profile_thread = threading.Thread(
            target=profile_printer, name="profile_printer", daemon=True,
        )
        profile_thread.start()

    frame_idx = 0
    t_start = time.time()
    try:
        while True:
            if pending_first_pair is not None:
                fa, fb = pending_first_pair
                pending_first_pair = None
            else:
                t_dec0 = time.perf_counter()
                ok, fa, fb = prefetch_reader.read()
                if prof is not None:
                    prof["decode"].record(
                        (time.perf_counter() - t_dec0) * 1000
                    )
                if not ok:
                    break
            t_put0 = time.perf_counter()
            compute_in_q.put((fa, fb, frame_idx))
            if prof is not None:
                prof["main_put_wait"].record(
                    (time.perf_counter() - t_put0) * 1000
                )
            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        compute_in_q.put(SENTINEL)
        compute_thread.join()
        composite_thread.join()
        yolo_q.put(SENTINEL)
        yolo_thread.join()
        if motion_thread is not None:
            motion_q.put(SENTINEL)
            motion_thread.join()
        prof_stop.set()
        if profile_thread is not None:
            profile_thread.join()
        prefetch_reader.close()
        writer.close()

    if worker_error[0] is not None:
        raise worker_error[0]

    elapsed = time.time() - t_start
    print()
    print(f"[info] Processed {frame_idx} frames in {elapsed:.2f}s "
          f"({frame_idx / max(elapsed, 1e-6):.2f} fps) "
          f"-- pipelined (compute + composite + yolo on separate threads)")
    print(sync_reader.summary_post())
    print(f"[info] Output written to {args.output}")

    if prof is not None:
        _print_profile(prof, "final profile (over entire run)")

    cap_a.release()
    cap_b.release()
