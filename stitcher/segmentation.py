"""
YOLO-based segmentation: people (for the seam-avoid mask) and static
foreground objects (chairs, couches, etc. — also for the seam-avoid
mask). Two backends:
    * YOLOv8 (fixed COCO 80-class set, fast)
    * YOLOE (open-vocabulary, text-prompted, slower but more accurate)

The same segmenter class wraps both; pick which one via the use_yoloe
constructor arg.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from stitcher.warp import dilate_gpu, warp_mask_gpu


PERSON_CLASS_ID = 0   # COCO 'person' class id, also the convention we use
                      # for YOLOE when person is the first text class.

# Default COCO classes for static foreground: furniture and large electronics.
# 56=chair, 57=couch, 59=bed, 60=dining table, 62=tv, 63=laptop, 73=book.
DEFAULT_FG_CLASS_IDS = [56, 57, 59, 60, 62, 63, 73]


class PersonSegmenter:
    """
    Thin wrapper over ultralytics.YOLO / ultralytics.YOLOE.

    When use_yoloe=True, text_classes must be a non-empty list of class
    name strings; the model is initialized with set_classes() and will
    return masks for those text-prompted classes only (in the list's
    order — class 0 is the first name, class 1 the second, etc.).
    """

    def __init__(self, weights_path: str, device: str = "cpu",
                 use_yoloe: bool = False, text_classes=None):
        try:
            if use_yoloe:
                from ultralytics import YOLOE
                if not text_classes:
                    raise RuntimeError("YOLOE requires a non-empty text_classes list.")
                self.model = YOLOE(weights_path)
                self.model.set_classes(
                    list(text_classes),
                    self.model.get_text_pe(list(text_classes)),
                )
            else:
                from ultralytics import YOLO
                self.model = YOLO(weights_path)
        except ImportError as e:
            raise RuntimeError("pip install ultralytics") from e
        self.device = device
        try:
            self.model.to(device)
        except Exception as e:
            print(f"[yolo] Could not move model to {device}: {e}.")

    def predict_classes_mask(self, frame_bgr, class_ids=(PERSON_CLASS_ID,)):
        """
        CPU/numpy variant. Returns a (H, W) uint8 numpy mask (0 or 255)
        with the union of all detected instances of any class in
        `class_ids`.
        """
        H, W = frame_bgr.shape[:2]
        results = self.model.predict(
            frame_bgr, classes=list(class_ids),
            verbose=False, retina_masks=False,
            device=self.device,
        )
        mask = np.zeros((H, W), dtype=np.uint8)
        if not results:
            return mask
        r = results[0]
        if r.masks is None or r.masks.data is None or len(r.masks.data) == 0:
            return mask
        mdata = r.masks.data.cpu().numpy()
        merged_small = (mdata > 0.5).any(axis=0).astype(np.uint8) * 255
        return cv2.resize(merged_small, (W, H), interpolation=cv2.INTER_NEAREST)

    def predict_classes_mask_gpu(self, frame_bgr, target_hw,
                                 class_ids=(PERSON_CLASS_ID,)):
        """
        GPU variant. Returns a (H, W) uint8 tensor on GPU with the union
        of all detected instances of any class in `class_ids`.
        """
        H_tgt, W_tgt = target_hw
        results = self.model.predict(
            frame_bgr, classes=list(class_ids),
            verbose=False, retina_masks=False,
            device=self.device,
        )
        if not results:
            return torch.zeros((H_tgt, W_tgt), dtype=torch.uint8,
                               device=self.device)
        r = results[0]
        if r.masks is None or r.masks.data is None or len(r.masks.data) == 0:
            return torch.zeros((H_tgt, W_tgt), dtype=torch.uint8,
                               device=self.device)
        mdata = r.masks.data
        merged = (mdata > 0.5).any(dim=0).float()
        m = merged.unsqueeze(0).unsqueeze(0)
        m = F.interpolate(m, size=(H_tgt, W_tgt), mode="nearest")
        mask = (m[0, 0] * 255).to(torch.uint8)
        return mask

    def predict_classes_mask_pair_gpu(self, frame_a_bgr, frame_b_bgr,
                                      target_hw_a, target_hw_b,
                                      class_ids=(PERSON_CLASS_ID,)):
        """
        Two-frame inference. When both frames share a shape (the common
        case) we run ONE model.predict over the list and let ultralytics
        batch them — saves a chunk of per-call overhead. When shapes
        differ, ultralytics has to letterbox both frames to a common
        batched shape, and the model masks come back at the model's
        grid resolution (e.g. 160x160 for a 640 input) covering that
        letterboxed extent including padding. F.interpolate'ing those
        masks back to each camera's NATIVE shape then stretches the
        padded area together with the valid area, distorting the mask
        (visible as a "flattened" person mask in the debug overlay).
        Fall back to two separate predict() calls in that case — each
        frame gets its own letterboxing and the masks line up with
        their native shapes.
        """
        if frame_a_bgr.shape == frame_b_bgr.shape:
            return self._predict_classes_mask_pair_batched_gpu(
                frame_a_bgr, frame_b_bgr, target_hw_a, target_hw_b, class_ids,
            )
        mask_a = self.predict_classes_mask_gpu(
            frame_a_bgr, target_hw_a, class_ids,
        )
        mask_b = self.predict_classes_mask_gpu(
            frame_b_bgr, target_hw_b, class_ids,
        )
        return mask_a, mask_b

    def _predict_classes_mask_pair_batched_gpu(self, frame_a_bgr, frame_b_bgr,
                                               target_hw_a, target_hw_b,
                                               class_ids):
        """Fast path used when both frames have the same shape — see
        predict_classes_mask_pair_gpu for the dispatch rationale."""
        results = self.model.predict(
            [frame_a_bgr, frame_b_bgr], classes=list(class_ids),
            verbose=False, retina_masks=False,
            device=self.device,
        )
        out = []
        for i, target_hw in enumerate((target_hw_a, target_hw_b)):
            H_tgt, W_tgt = target_hw
            if not results or i >= len(results):
                out.append(torch.zeros((H_tgt, W_tgt), dtype=torch.uint8,
                                       device=self.device))
                continue
            r = results[i]
            if r.masks is None or r.masks.data is None or len(r.masks.data) == 0:
                out.append(torch.zeros((H_tgt, W_tgt), dtype=torch.uint8,
                                       device=self.device))
                continue
            mdata = r.masks.data
            merged = (mdata > 0.5).any(dim=0).float()
            m = merged.unsqueeze(0).unsqueeze(0)
            m = F.interpolate(m, size=(H_tgt, W_tgt), mode="nearest")
            out.append((m[0, 0] * 255).to(torch.uint8))
        return out[0], out[1]

    def predict_classes_mask_pair(self, frame_a_bgr, frame_b_bgr,
                                  class_ids=(PERSON_CLASS_ID,)):
        """CPU/numpy variant of predict_classes_mask_pair_gpu — same
        same-shape-batched / mismatched-fallback dispatch."""
        if frame_a_bgr.shape == frame_b_bgr.shape:
            return self._predict_classes_mask_pair_batched(
                frame_a_bgr, frame_b_bgr, class_ids,
            )
        mask_a = self.predict_classes_mask(frame_a_bgr, class_ids)
        mask_b = self.predict_classes_mask(frame_b_bgr, class_ids)
        return mask_a, mask_b

    def _predict_classes_mask_pair_batched(self, frame_a_bgr, frame_b_bgr,
                                           class_ids):
        results = self.model.predict(
            [frame_a_bgr, frame_b_bgr], classes=list(class_ids),
            verbose=False, retina_masks=False,
            device=self.device,
        )
        out = []
        for i, frame_bgr in enumerate((frame_a_bgr, frame_b_bgr)):
            H, W = frame_bgr.shape[:2]
            mask = np.zeros((H, W), dtype=np.uint8)
            if results and i < len(results):
                r = results[i]
                if r.masks is not None and r.masks.data is not None \
                        and len(r.masks.data) > 0:
                    mdata = r.masks.data.cpu().numpy()
                    merged_small = (mdata > 0.5).any(axis=0).astype(np.uint8) * 255
                    mask = cv2.resize(merged_small, (W, H),
                                      interpolation=cv2.INTER_NEAREST)
            out.append(mask)
        return out[0], out[1]


def compute_fg_mask_seg_gpu(segmenter, frame_a, frame_b, class_ids,
                             grid_a_t, grid_b_t, dilate_radius,
                             overlap_bbox, overlap_in_bbox_t):
    """
    Static foreground mask via instance segmentation (GPU).
    Runs the segmenter on each ORIGINAL frame asking for `class_ids`,
    warps each mask to the canvas via grid_sample, unions, dilates,
    and crops to the overlap bbox.

    Returns a (H_bb, W_bb) uint8 tensor on GPU (0 or 255).
    """
    H_a, W_a = frame_a.shape[:2]
    H_b, W_b = frame_b.shape[:2]

    mask_a_src_t = segmenter.predict_classes_mask_gpu(frame_a, (H_a, W_a), class_ids)
    mask_b_src_t = segmenter.predict_classes_mask_gpu(frame_b, (H_b, W_b), class_ids)

    mask_a_canvas_t = warp_mask_gpu(mask_a_src_t, grid_a_t)
    mask_b_canvas_t = warp_mask_gpu(mask_b_src_t, grid_b_t)

    union_t = torch.bitwise_or(mask_a_canvas_t, mask_b_canvas_t)
    union_t = dilate_gpu(union_t, dilate_radius)

    x0, y0, x1, y1 = overlap_bbox
    fg_mask_bbox_t = union_t[y0:y1, x0:x1].contiguous()
    # AND with overlap shape inside bbox (from passed overlap_in_bbox_t,
    # which is the bbox-sized version).
    fg_mask_bbox_t = torch.where(overlap_in_bbox_t > 0,
                                 fg_mask_bbox_t,
                                 torch.zeros_like(fg_mask_bbox_t))
    return fg_mask_bbox_t


def compute_fg_mask_seg_cpu(segmenter, frame_a, frame_b, class_ids,
                             map_ax, map_ay, map_bx, map_by,
                             fg_dilate_kernel, overlap_bbox, overlap_in_bbox):
    """
    CPU variant of compute_fg_mask_seg_gpu. Returns a (H_bb, W_bb) uint8
    numpy mask (0 or 255), cropped to the overlap bbox and AND'd with
    the overlap shape.
    """
    mask_a_src = segmenter.predict_classes_mask(frame_a, class_ids)
    mask_b_src = segmenter.predict_classes_mask(frame_b, class_ids)
    mask_a_canvas = cv2.remap(mask_a_src, map_ax, map_ay, cv2.INTER_NEAREST)
    mask_b_canvas = cv2.remap(mask_b_src, map_bx, map_by, cv2.INTER_NEAREST)
    union = cv2.bitwise_or(mask_a_canvas, mask_b_canvas)
    if fg_dilate_kernel is not None:
        union = cv2.dilate(union, fg_dilate_kernel)
    x0, y0, x1, y1 = overlap_bbox
    fg_bbox = union[y0:y1, x0:x1].copy()
    return cv2.bitwise_and(fg_bbox, overlap_in_bbox)
