"""
VLM judge: pick which detected classes the panorama seam should avoid.

Step C of the auto-FG-v2 discovery flow. Given:
  - one camera frame
  - the depth-annotated inventory from
    stitcher.depth_inventory.annotate_inventory_with_depth

ask a locally-hosted vision-language model (Qwen2.5-VL via Ollama by
default) to judge which classes in the inventory are visually
important enough that the stitching seam should route around them.

The big change from auto_fg v1 is that the VLM ranks by *importance*,
not by foreground/background. A TV mounted on the back wall is far
in depth but visually critical; a rug in the foreground is close in
depth but the seam is invisible through it. The depth value goes
into the prompt as a hint, but the VLM is the final decider.

Why single-image, single-message: Qwen2.5-VL on Ollama has been
unreliable with multi-image or composite inputs (GGML asserts, raw
chat-template tokens leaking into the output). One frame in one
message is the only call shape that consistently behaves.
"""

import cv2


DEFAULT_OLLAMA_MODEL = "qwen2.5vl:3b"


# Image is downscaled before being sent to the VLM so the call fits
# comfortably on a T1000 (8 GB). Long-side limit at 896 keeps the
# token count low without losing recognisable objects.
_VLM_MAX_DIM = 896


# The judge prompt. Edit this directly to tune which kinds of object
# the VLM is inclined to keep. The {class_summary} placeholder is
# filled per call from the depth-annotated inventory.
JUDGE_PROMPT_TEMPLATE = """\
You are helping configure a real-time video stitching system. A
panorama seam will be drawn between two camera views of the same
room. The seam can cause visible distortion where it crosses an
object, so we need to route it AROUND a small set of visually
important objects.

Look at the image and at the list of detected objects. Among the
list, keep only the classes that the seam should avoid cutting
through.

Each entry lists how many instances of the class were detected and
their depth range. Depth is normalized: 0.0 = far from the camera,
1.0 = close to the camera.

Detected objects:
{class_summary}

Rules for picking classes to protect:
  - Pick classes that would look bad if a seam cut across them:
    items with sharp edges, text or screens, items the eye naturally
    tracks.
  - Foreground objects are more important, because their 
    low distance to the cameras increase the parallax. Thus, a table
    in the background must be dropped, while a table in the foreground
    must be kept for example
  - Far objects matter only if they are visually important. A TV on 
    the back wall is worth protecting even though it is far, while a
    plant on the back wall is not.
  - EXCLUDE:
      - floor or wall coverings that blend with surroundings
        (carpets, rugs, mats, plain panels)
      - things so small or visually flat that a seam through them
        is invisible
      - structural background (walls, ceilings, doors, windows)
  - 3 to 8 classes total.

Output format:
  - ONLY a comma-separated list. No prose, no preface, no period,
    no markdown.
  - Class names must come EXACTLY from the list above, in lowercase.

Example output: chair, tv, desk, picture frame

Concrete examples:
  - A blue chair on the foreground -> INCLUDE (it can cause visual 
    artefacts because of the high parallax)
  - A red chair on the background -> EXCLUDE (it is on the background
    and is not an important object)
  - A frame on the background -> INCLUDE (it is on the background
    but is a notable object that should be preserved in the panorama)
  - A ceiling light -> EXCLUDE (ceiling lights are structural and not
    visually important)
  - A plant on the background -> EXCLUDE (plants are visually complex
    but often blend well with the surroundings, plus it's not the
    main focus of the scene)

"""


def _encode_png_bytes(frame_bgr):
    ok, buf = cv2.imencode(".png", frame_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed on judge frame")
    return buf.tobytes()


def _downscale_long_side(frame_bgr, max_dim):
    H, W = frame_bgr.shape[:2]
    longest = max(H, W)
    if longest <= max_dim:
        return frame_bgr
    scale = max_dim / longest
    new_w = max(1, int(round(W * scale)))
    new_h = max(1, int(round(H * scale)))
    return cv2.resize(frame_bgr, (new_w, new_h),
                      interpolation=cv2.INTER_AREA)


def _summarize_inventory(records):
    """
    Group depth-annotated records by class name; return a multiline
    bullet list with count + depth range for each class.

    The order is by max depth descending so the VLM reads the
    closest classes first -- a soft prior that closer things tend
    to matter more, without forcing it on the model.
    """
    by_class = {}
    for r in records:
        by_class.setdefault(r["class"], []).append(r["depth"])

    rows = []
    for name, depths in by_class.items():
        d_min = min(depths)
        d_max = max(depths)
        if len(depths) == 1:
            depth_str = f"depth {d_max:.2f}"
        else:
            depth_str = f"depth {d_min:.2f} - {d_max:.2f}"
        rows.append((d_max, name, len(depths), depth_str))

    rows.sort(key=lambda t: t[0], reverse=True)
    lines = []
    for _, name, n, depth_str in rows:
        plural = "instance" if n == 1 else "instances"
        lines.append(f"  - {name} ({n} {plural}, {depth_str})")
    return "\n".join(lines)


def _parse_class_list(text, allowed):
    """
    Parse the VLM's response into a list of class names, keeping only
    those that appeared in the inventory (case-insensitive match).
    Tolerates leading/trailing punctuation, markdown, prose noise.
    """
    text = (text or "").strip().strip(".").strip()
    text = text.replace("**", "").replace("__", "").replace("`", "")
    for prefix in ("output:", "answer:", "classes:", "result:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
    items = [it.strip().strip('"').strip("'").lower()
             for it in text.split(",")]
    items = [it for it in items if it]
    allowed_lc = {a.lower(): a for a in allowed}
    seen = set()
    out = []
    for it in items:
        if it in allowed_lc and it not in seen:
            seen.add(it)
            out.append(allowed_lc[it])
    return out


def judge_inventory(frame_bgr, records,
                    model_name=DEFAULT_OLLAMA_MODEL,
                    prompt_template=JUDGE_PROMPT_TEMPLATE,
                    return_raw=False):
    """
    Ask the VLM which classes in `records` are important enough that
    the panorama seam should avoid cutting through them.

    Args:
        frame_bgr: HxWx3 uint8 BGR image -- the SINGLE frame the VLM
            looks at (typically the left camera). Multi-image and
            composite inputs have been unreliable, so the call stays
            single-frame.
        records: list of dicts as returned by
            stitcher.depth_inventory.annotate_inventory_with_depth.
        model_name: Ollama tag of a Qwen2.5-VL model. Default is the
            3 B variant which fits in 8 GB VRAM. Pass a larger tag if
            you have the headroom.
        prompt_template: edit JUDGE_PROMPT_TEMPLATE at module scope
            to tune what the VLM keeps, or pass an override here.
            Must contain a {class_summary} placeholder.
        return_raw: if True, also return the VLM's raw text response
            (useful for the validator).

    Returns:
        list of class names to protect (subset of the classes
        appearing in `records`, preserving the VLM's order).
        If return_raw is True, returns (classes, raw_text).

    Raises RuntimeError with an actionable hint when the `ollama`
    package is missing, the daemon isn't running, the model isn't
    pulled, or the VLM returned no usable class names.
    """
    if not records:
        if return_raw:
            return [], ""
        return []

    try:
        import ollama
    except ImportError as e:
        raise RuntimeError(
            "VLM judge requires the `ollama` Python package. Install with:\n"
            "    pip install ollama"
        ) from e

    class_summary = _summarize_inventory(records)
    prompt = prompt_template.format(class_summary=class_summary)

    small = _downscale_long_side(frame_bgr, _VLM_MAX_DIM)
    img_png = _encode_png_bytes(small)

    try:
        response = ollama.chat(
            model=model_name,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [img_png],
            }],
            options={
                # Deterministic: same room, same answer.
                "temperature": 0.0,
            },
        )
    except Exception as e:
        raise RuntimeError(
            f"Ollama call failed: {e}\n"
            "Common causes:\n"
            "  - Ollama daemon not running. Start the Ollama app or "
            "run `ollama serve`.\n"
            f"  - Model not pulled. Run `ollama pull {model_name}`.\n"
            "  - Different tag on your machine. Pass model_name=... "
            "to override."
        ) from e

    raw = response.get("message", {}).get("content", "")
    allowed = sorted({r["class"] for r in records})
    classes = _parse_class_list(raw, allowed)
    if not classes:
        raise RuntimeError(
            "VLM returned no usable class names. Raw response was:\n"
            f"---\n{raw}\n---"
        )
    if return_raw:
        return classes, raw
    return classes
