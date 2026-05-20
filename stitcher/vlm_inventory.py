"""
VLM-generated inventory vocabulary.

Step 0 of the auto-FG-v2 discovery flow: ask the VLM to look at one
camera frame and list every object in the room, with colors and
other distinguishing details. The output is a list of short noun
phrases (e.g. "blue armchair", "wooden desk", "framed mountain
poster", "tall green houseplant") that gets fed straight into YOLOE
as its open-vocabulary text-class list.

Replaces the hardcoded stitcher.object_inventory.INVENTORY_CLASSES
default: instead of asking YOLOE to look for ~40 generic classes,
we ask YOLOE to look for the specific items the VLM saw in this
room. Sharper YOLOE detections, and the downstream judge sees
specific class names ("blue armchair" vs "gray office chair")
instead of generic ones ("chair").

Same single-image / single-message Ollama call shape used by
stitcher.vlm_judge -- the only call pattern that's been reliable on
T1000-class GPUs with Qwen2.5-VL.
"""

import cv2


DEFAULT_OLLAMA_MODEL = "qwen2.5vl:3b"


# Image is downscaled before being sent to the VLM. Long-side 896
# keeps token count low without losing recognisable objects.
_VLM_MAX_DIM = 896


# Module-level so callers can edit. Tune what kinds of detail the
# VLM emphasises (color, material, content of screens, etc.) by
# editing this template.
INVENTORY_PROMPT = """\
You are helping configure a real-time video stitching system. The
attached image shows one of two camera views of a room. I need a
complete inventory of the objects in the room -- furniture, decor,
electronics, plants, anything notable. The list will be fed to an
open-vocabulary object detector (YOLOE) which will then locate
each item in the video.

Rules:
  - Output ONLY a comma-separated list. No prose, no preface, no
    period at the end, no markdown, no numbering.
  - Each item is a short noun phrase, 1 to 4 words.
  - Be SPECIFIC about color, material, or distinguishing details
    when they would help identify the object:
        "blue armchair" beats "chair"
        "wooden desk" beats "desk"
        "framed mountain poster" beats "poster"
        "dual monitor setup" beats "monitor"
        "tall green houseplant" beats "plant"
  - List multiple distinct items of the same kind SEPARATELY:
        "blue armchair, gray office chair, wooden chair"
    (not "three chairs").
  - INCLUDE: furniture, electronics, lamps, plants, picture frames,
    posters, screens, books, rugs, baskets, boxes, whiteboards,
    anything that takes up visual space.
  - EXCLUDE: people, hands, pets, walls, ceiling, floor, doors,
    windows, ceiling lights mounted to the structure.

Example output:
blue armchair, gray office chair, wooden desk, dual monitor setup, framed mountain poster, tall green houseplant, floor lamp with white shade, beige rug, whiteboard

Now list the objects in the attached room image:
"""


def _encode_png_bytes(frame_bgr):
    ok, buf = cv2.imencode(".png", frame_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed on inventory frame")
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


def _parse_phrase_list(text):
    """
    Parse the VLM's response into a list of short noun phrases.
    Strips markdown / numbering / surrounding punctuation, lowercases,
    deduplicates while preserving order. Drops phrases longer than 6
    words (defence against the VLM returning a sentence instead of
    a list).
    """
    text = (text or "").strip().strip(".").strip()
    text = text.replace("**", "").replace("__", "").replace("`", "")
    for prefix in ("output:", "answer:", "objects:", "result:",
                   "inventory:", "list:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
    # Some models like to put each item on its own line with a
    # bullet. Normalize: turn newlines + bullets into commas.
    text = text.replace("\n", ", ")
    for bullet in ("- ", "* ", "• "):
        text = text.replace(bullet, "")

    import re
    items = []
    for raw_item in text.split(","):
        s = raw_item.strip().strip('"').strip("'").lower()
        # Strip leading "1. ", "12) ", "iii. " etc. -- numbered-list
        # leakage when the VLM ignored the "no numbering" rule.
        s = re.sub(r"^[\divx]+[\.\)]\s+", "", s)
        if s:
            items.append(s)
    # Defence: drop anything implausibly long for a noun phrase.
    items = [it for it in items if 1 <= len(it.split()) <= 6]
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def list_objects_with_details(frame_bgr,
                              model_name=DEFAULT_OLLAMA_MODEL,
                              prompt=INVENTORY_PROMPT,
                              return_raw=False):
    """
    Ask the VLM for a per-scene inventory vocabulary.

    Args:
        frame_bgr: HxWx3 uint8 BGR image -- one camera frame
            (typically the left camera). One image, one message --
            the only call shape reliable with Qwen2.5-VL on Ollama.
        model_name: Ollama tag. Default qwen2.5vl:3b.
        prompt: edit INVENTORY_PROMPT at module scope to tune what
            detail the VLM emits.
        return_raw: if True, also return the VLM's raw text response.

    Returns:
        list of short noun phrases (e.g. ["blue armchair", "wooden
        desk", "framed mountain poster", ...]) -- the YOLOE
        text-class list for this scene.
        If return_raw is True: (phrases, raw_text).

    Raises RuntimeError with an actionable hint when `ollama` is
    missing, the daemon is unreachable, or the VLM returned no
    usable phrases.
    """
    try:
        import ollama
    except ImportError as e:
        raise RuntimeError(
            "VLM inventory requires the `ollama` Python package. "
            "Install with: pip install ollama"
        ) from e

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
                # Deterministic-ish: same room -> same inventory.
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
    phrases = _parse_phrase_list(raw)
    if not phrases:
        raise RuntimeError(
            "VLM returned no usable inventory phrases. Raw response:\n"
            f"---\n{raw}\n---"
        )
    if return_raw:
        return phrases, raw
    return phrases
