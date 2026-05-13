"""
End-to-end stitching pipeline.

`run(args)` takes a parsed argparse.Namespace, sets up the device /
geometry / segmenters / writer, then runs the per-frame loop. The
entry point script (`video_stitcher_seam_gpu.py`) owns the argparse;
this module owns the actual work.
"""

import os
import queue
import threading
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from stitcher.compositing import (
    composite_multiband_cpu,
    composite_multiband_gpu_resident,
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
    warp_gpu,
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

    crop_rect = None
    if args.autocrop:
        crop_rect = find_autocrop_rect(
            H_b_to_a, frame_a.shape, frame_b.shape, canvas_size, T,
        )
        cx, cy, cw, ch = crop_rect
        print(f"[info] Autocrop: x={cx} y={cy} size={cw}x{ch} "
              f"(from full canvas {canvas_size[0]}x{canvas_size[1]})")

    print("[info] Precomputing remap maps + static geometry...")
    map_ax, map_ay = build_remap(H_a_to_canvas, canvas_size)
    map_bx, map_by = build_remap(H_b_to_canvas, canvas_size)
    static = build_static_geometry(
        frame_a.shape, frame_b.shape,
        map_ax, map_ay, map_bx, map_by,
        canvas_size,
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
        H_canvas, W_canvas = canvas_size[1], canvas_size[0]
        gpu_ctx = {
            "device": torch_device,
            "kernel2d": get_pyr_kernel_2d(torch_device),
            "only_a_u8_t": torch.from_numpy(static["only_a_u8"]).to(torch_device),
            "only_b_u8_t": torch.from_numpy(static["only_b_u8"]).to(torch_device),
            "only_a_in_bbox_t": torch.from_numpy(static["only_a_in_bbox"]).to(torch_device),
            "only_b_in_bbox_t": torch.from_numpy(static["only_b_in_bbox"]).to(torch_device),
            "overlap_in_bbox_t": torch.from_numpy(static["overlap_in_bbox"]).to(torch_device),
            "valid_in_bbox_t": torch.from_numpy(valid_in_bbox_np).to(torch_device),
            # Page-locked host buffer for the final GPU->CPU transfer of
            # the composited frame; lets CUDA's DMA copy bypass an extra
            # staging copy. Allocated once for the full canvas.
            "pinned_output_t": torch.empty(
                (H_canvas, W_canvas, 3),
                dtype=torch.uint8, pin_memory=True,
            ),
        }
        grid_a_t = build_grid_sample_tensor(map_ax, map_ay, frame_a.shape, torch_device)
        grid_b_t = build_grid_sample_tensor(map_bx, map_by, frame_b.shape, torch_device)
        # Stacked grid for the batched warp_pair_gpu call. Built once
        # since the grids are static for the whole run.
        grid_pair_t = torch.cat([grid_a_t, grid_b_t], dim=0)
        overlap_in_bbox_t = torch.from_numpy(static["overlap_in_bbox"]).to(torch_device)
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

    W, H = canvas_size
    out_buf      = np.zeros((H, W, 3), dtype=np.uint8)
    cost_scratch = np.empty((bbox_shape[0], bbox_shape[1], 3), dtype=np.float32)
    person_mask_bbox = np.zeros(bbox_shape, dtype=np.uint8)
    person_mask_bbox_t = None
    if dev["cuda_available"]:
        person_mask_bbox_t = torch.zeros(bbox_shape, dtype=torch.uint8,
                                         device=gpu_ctx["device"])
    # Float EMA buffers for person mask temporal smoothing. Filled on the
    # first YOLO run; None means "no history yet".
    person_mask_ema_t = None  # GPU path
    person_mask_ema = None    # CPU path
    cost_ema = None
    seam_prev_small = None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_size = (crop_rect[2], crop_rect[3]) if crop_rect else canvas_size
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
        """Warp + (every yolo_every) YOLO + mask + cost + EMA + DP seam.
        Updates inter-frame state in the enclosing scope. Returns a payload
        dict consumed by composite_one."""
        nonlocal cost_ema_t, cost_ema
        nonlocal person_mask_bbox_t, person_mask_bbox
        nonlocal person_mask_ema_t, person_mask_ema
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

        # YOLO person mask (every yolo_every frames). Stateful EMA on the
        # binary mask between runs.
        if frame_idx % args.yolo_every == 0:
            if dev["cuda_available"]:
                # Batched YOLO call for both cameras (one model.predict
                # over the stacked pair instead of two separate calls).
                mask_a_src_t, mask_b_src_t = (
                    person_segmenter.predict_classes_mask_pair_gpu(
                        frame_a, frame_b,
                        frame_a.shape[:2], frame_b.shape[:2],
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
                if args.debug_mask:
                    person_mask_bbox = person_mask_bbox_t.cpu().numpy()
            else:
                mask_a_src, mask_b_src = (
                    person_segmenter.predict_classes_mask_pair(
                        frame_a, frame_b, person_class_ids,
                    )
                )
                mask_a_canvas = cv2.remap(mask_a_src, map_ax, map_ay, cv2.INTER_NEAREST)
                mask_b_canvas = cv2.remap(mask_b_src, map_bx, map_by, cv2.INTER_NEAREST)
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
            if use_fg and fg_mask_bbox is not None:
                if person_mask_bbox.any():
                    fg_only = (fg_mask_bbox > 0) & (person_mask_bbox == 0)
                else:
                    fg_only = fg_mask_bbox > 0
                cost_for_dp[fg_only] += args.fg_penalty
            if person_mask_bbox.any():
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

        # Snapshot the FG mask for the debug overlay (composite stage may
        # run a frame later, so we capture the version that's current
        # for THIS frame here).
        fg_for_debug = None
        if args.debug_mask and use_fg:
            if fg_mask_bbox_t is not None:
                fg_for_debug = fg_mask_bbox_t.cpu().numpy()
            elif fg_mask_bbox is not None:
                fg_for_debug = fg_mask_bbox

        person_for_debug = person_mask_bbox if args.debug_mask else None

        return {
            "frame_idx": frame_idx,
            "warped_a_t": warped_a_t,
            "warped_b_t": warped_b_t,
            "warped_a": warped_a,
            "warped_b": warped_b,
            "seam_x_full": seam_x_full,
            "person_for_debug": person_for_debug,
            "fg_for_debug": fg_for_debug,
        }

    def composite_one(payload):
        """Run the multi-band composite, debug overlays, autocrop, and
        enqueue the final frame to the writer."""
        seam_x_full = payload["seam_x_full"]
        if gpu_ctx is not None:
            stitched = composite_multiband_gpu_resident(
                payload["warped_a_t"], payload["warped_b_t"],
                static, seam_x_full,
                args.blend_width, args.blend_levels, out_buf,
                gpu_ctx,
            )
        else:
            stitched = composite_multiband_cpu(
                payload["warped_a"], payload["warped_b"],
                static, seam_x_full,
                args.blend_width, args.blend_levels, out_buf,
            )
        if args.debug_mask:
            fg_for_debug = payload["fg_for_debug"]
            if fg_for_debug is not None:
                draw_mask_overlay(stitched, fg_for_debug,
                                  static["overlap_bbox"],
                                  color=(0, 255, 255), alpha=0.25)
            person_for_debug = payload["person_for_debug"]
            if person_for_debug is not None:
                draw_mask_overlay(stitched, person_for_debug,
                                  static["overlap_bbox"])
        if args.debug_seam:
            draw_seam_overlay(stitched, seam_x_full, static["overlap_bbox"])
        if crop_rect is not None:
            cx, cy, cw, ch = crop_rect
            stitched = stitched[cy:cy + ch, cx:cx + cw]
        return stitched

    # --- Pipelined execution with two worker threads ---------------------
    SENTINEL = object()
    compute_in_q = queue.Queue(maxsize=4)
    composite_in_q = queue.Queue(maxsize=4)
    worker_error = [None]

    def compute_worker():
        try:
            while True:
                item = compute_in_q.get()
                if item is SENTINEL:
                    composite_in_q.put(SENTINEL)
                    return
                fa, fb, idx = item
                payload = compute_one(fa, fb, idx)
                composite_in_q.put(payload)
        except Exception as e:
            worker_error[0] = e
            composite_in_q.put(SENTINEL)

    def composite_worker():
        try:
            while True:
                item = composite_in_q.get()
                if item is SENTINEL:
                    return
                stitched = composite_one(item)
                writer.write(stitched)
        except Exception as e:
            worker_error[0] = e

    compute_thread = threading.Thread(
        target=compute_worker, name="compute", daemon=True,
    )
    composite_thread = threading.Thread(
        target=composite_worker, name="composite", daemon=True,
    )
    compute_thread.start()
    composite_thread.start()

    frame_idx = 0
    t_start = time.time()
    try:
        while True:
            if pending_first_pair is not None:
                fa, fb = pending_first_pair
                pending_first_pair = None
            else:
                ok, fa, fb = prefetch_reader.read()
                if not ok:
                    break
            compute_in_q.put((fa, fb, frame_idx))
            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        compute_in_q.put(SENTINEL)
        compute_thread.join()
        composite_thread.join()
        prefetch_reader.close()
        writer.close()

    if worker_error[0] is not None:
        raise worker_error[0]

    elapsed = time.time() - t_start
    print()
    print(f"[info] Processed {frame_idx} frames in {elapsed:.2f}s "
          f"({frame_idx / max(elapsed, 1e-6):.2f} fps) "
          f"-- pipelined (compute + composite on separate threads)")
    print(sync_reader.summary_post())
    print(f"[info] Output written to {args.output}")

    cap_a.release()
    cap_b.release()
