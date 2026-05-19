"""
Depth estimation helper used by the depth-aware variant of the
auto-FG discovery flow.

Loads Depth Anything V2 (Small) via Hugging Face transformers on
first call and keeps the pipeline cached in process memory so the
second frame of a discovery run reuses the already-loaded model.

Lazy import: `transformers` is only required when depth filtering
is actually used. The rest of the pipeline doesn't need it.

Output is INVERSE depth (larger value = closer to the camera) at
the same (H, W) shape as the input frame, so subsequent bbox-
median lookups index pixel coords directly.
"""

import cv2
import numpy as np


_DEPTH_PIPE = None  # process-cached HF pipeline


def _get_depth_pipe():
    """Lazy-load the Depth Anything V2 (Small) pipeline."""
    global _DEPTH_PIPE
    if _DEPTH_PIPE is not None:
        return _DEPTH_PIPE
    try:
        from transformers import pipeline
    except ImportError as e:
        raise RuntimeError(
            "depth-aware --depth_threshold requires the "
            "`transformers` Python package. Install with:\n"
            "    pip install transformers"
        ) from e
    _DEPTH_PIPE = pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
    )
    return _DEPTH_PIPE


def estimate_depth(frame_bgr):
    """
    Return a (H, W) float32 inverse-depth map for an OpenCV BGR
    frame. Larger value = closer to the camera.

    Depth Anything V2 internally accepts a PIL RGB image and returns
    a result dict containing the depth as a PIL image. We convert to
    numpy and resize to the input frame's resolution if the model
    happened to emit a downsampled tensor, so a (y, x) bbox lookup
    indexes the right pixel.

    First call is slow (~2-5 s model load on a T1000); subsequent
    calls in the same process are ~100-300 ms on 1080p frames.
    """
    from PIL import Image

    pipe = _get_depth_pipe()
    H, W = frame_bgr.shape[:2]
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    result = pipe(img)
    depth_pil = result["depth"]
    depth = np.array(depth_pil, dtype=np.float32)
    if depth.shape != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
    return depth
