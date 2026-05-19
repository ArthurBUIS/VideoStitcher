"""
Auto-discovery of the static-FG class list for --yoloe_fg_classes.

Asks a locally-hosted vision-language model (Qwen2.5-VL via Ollama by
default) to look at one frame from each camera and propose the
prominent static foreground objects in the room. The output is fed
into YOLOE for per-frame segmentation, which then guides the seam
away from those objects.

Two entry points:
    * tools/suggest_fg_classes.py -- standalone CLI; prints a
      comma-separated list to stdout (suitable for piping into
      --yoloe_fg_classes).
    * --yoloe_fg_classes auto -- video_stitcher_seam_gpu.py picks this
      sentinel up in main() and runs suggest_fg_classes() before
      booting the pipeline, replacing args.yoloe_fg_classes in place.

The VLM call happens BEFORE any stitching state is allocated, so the
8 GB of VRAM on a T1000 isn't oversubscribed -- by the time the
stitcher initialises its YOLOE + PyTorch context, Ollama has
released the model.

The `ollama` Python package is a lazy import so the rest of the
pipeline doesn't need it installed unless auto-discovery is used.
"""

import cv2


DEFAULT_OLLAMA_MODEL = "qwen2.5vl:7b"


# Auto-discovery sentinel(s): if --yoloe_fg_classes is set to one of
# these (case-insensitive, exactly one item), main() runs the VLM
# instead of using the literal class name.
AUTO_SENTINELS = ("auto", "automatic")


# The few-shot prompt. Text-only examples (no image examples) because
# (a) embedding example images into the prompt makes the request
# heavier for the VLM and (b) the format is simple enough that text
# examples are sufficient to nail the output shape. The actual query
# attaches ONE side-by-side composite image (camera A on the left,
# camera B on the right) -- passing two images as separate entries in
# the same Ollama message confuses Qwen2.5-VL's chat template and
# makes it emit the raw <|im_start|> token instead of a real answer.
_PROMPT = """\
You are helping configure a real-time video stitching system. Your
job: look at a side-by-side image showing two camera views of the
SAME room (camera A on the left half, camera B on the right half)
and list the static foreground objects in it.

The class names you output will be fed to an open-vocabulary
segmentation model (YOLOE) which will detect them in every frame. The
seam-finding step then routes the panorama seam AROUND those objects
so the seam never cuts through them.

Rules:
  - Output ONLY a comma-separated list. NO other text, no header, no
    period at the end, no markdown.
  - 4 to 10 items total.
  - Each item should be a singular noun or a short noun phrase
    (1-3 words).
  - Be specific when distinctive: "yellow chair" beats "chair" if it
    is a notable colour; "monitor" beats "screen"; "stool" beats
    "seat".
  - Include: chairs, couches, tables, monitors, plants, picture
    frames, lamps, electronics, whiteboards, and other semi-permanent
    foreground objects that someone walking through the room would
    visually notice.
  - EXCLUDE: walls, floor, ceiling, doors, windows, ceiling lights,
    light fixtures mounted to the structure, people, pets, hands.
  - Treat the two halves as views of the same room -- an object
    visible in only one half still counts once. Output each item only
    once even if it appears in both halves.

Examples of correct outputs from OTHER rooms (these are just to show
the FORMAT -- do not copy them blindly):
    yellow chair, monitor, stool, picture frame, plant
    couch, coffee table, tv, bookshelf, lamp, ottoman
    desk, monitor, keyboard, office chair, plant, whiteboard

Now look at this side-by-side image of the target room and output
the class list:
"""


def _encode_png_bytes(frame_bgr):
    """OpenCV BGR uint8 array -> PNG bytes (what the Ollama HTTP
    image field expects)."""
    ok, buf = cv2.imencode(".png", frame_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed on baseline frame")
    return buf.tobytes()


# Per-frame longest-side limit before the side-by-side concat. With
# typical 1920x1080 cameras this brings each half down to about 900x506
# and the composite to roughly 1800x506 — well inside what Qwen2.5-VL
# tokenises efficiently on a T1000. The model downsamples internally
# anyway, so we lose no usable detail.
_VLM_PER_FRAME_MAX_DIM = 900


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


def _side_by_side(frame_a_bgr, frame_b_bgr,
                  per_frame_max_dim=_VLM_PER_FRAME_MAX_DIM):
    """Stack two BGR frames horizontally for the VLM.

    Each frame is first downscaled so its longest side is at most
    `per_frame_max_dim` — without this, a 1920x1080 + 1920x1080 pair
    becomes a 3840x1080 composite that produces thousands of visual
    tokens and hangs the Qwen2.5-VL inference on small GPUs.

    If the cameras have different heights after the per-frame
    downscale, the taller one is resized down to match the shorter so
    cv2.hconcat works.

    We pass a single combined image to Ollama because Qwen2.5-VL on
    Ollama mis-handles two images in the same chat message (emits the
    raw <|im_start|> chat-template token instead of a real answer).
    """
    a = _downscale_long_side(frame_a_bgr, per_frame_max_dim)
    b = _downscale_long_side(frame_b_bgr, per_frame_max_dim)
    H_a = a.shape[0]
    H_b = b.shape[0]
    if H_a != H_b:
        target_h = min(H_a, H_b)
        if H_a != target_h:
            new_w = max(1, int(round(a.shape[1] * target_h / H_a)))
            a = cv2.resize(a, (new_w, target_h), interpolation=cv2.INTER_AREA)
        if H_b != target_h:
            new_w = max(1, int(round(b.shape[1] * target_h / H_b)))
            b = cv2.resize(b, (new_w, target_h), interpolation=cv2.INTER_AREA)
    return cv2.hconcat([a, b])


def _parse_class_list(text):
    """Parse the VLM's response into a clean list of class names.

    Tolerates: leading/trailing whitespace, surrounding punctuation,
    Markdown bold/italics, mixed-case, duplicates. Returns lowercase
    names in the order they appeared.
    """
    text = text.strip().strip(".").strip()
    # Strip Markdown if any
    text = text.replace("**", "").replace("__", "").replace("`", "")
    # Drop any leading "Output:" / "Answer:" / etc. prefix
    for prefix in ("output:", "answer:", "classes:", "result:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
    items = [it.strip().strip('"').strip("'").lower()
             for it in text.split(",")]
    items = [it for it in items if it]
    # Deduplicate while preserving order.
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def suggest_fg_classes(frame_a_bgr, frame_b_bgr,
                       model_name=DEFAULT_OLLAMA_MODEL,
                       prompt=_PROMPT):
    """
    Query the local VLM with both camera frames + a few-shot prompt;
    return a list of static-FG class names to feed to YOLOE.

    Raises RuntimeError with an actionable hint when:
      - the `ollama` Python package isn't installed
      - the Ollama daemon isn't running
      - the requested model isn't pulled
      - the VLM returned no usable class names
    """
    try:
        import ollama
    except ImportError as e:
        raise RuntimeError(
            "auto-discovery of --yoloe_fg_classes requires the `ollama` "
            "Python package. Install it with: pip install ollama"
        ) from e

    combined_png = _encode_png_bytes(_side_by_side(frame_a_bgr, frame_b_bgr))

    try:
        response = ollama.chat(
            model=model_name,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [combined_png],
            }],
            options={
                # Deterministic-ish output; we want the same class list
                # if the same room is analysed twice. The few-shot prompt
                # constrains the format enough that temperature=0 is
                # safe.
                "temperature": 0.0,
            },
        )
    except Exception as e:
        raise RuntimeError(
            f"Ollama call failed: {e}\n"
            "Common causes:\n"
            "  - Ollama daemon not running. Start the Ollama app, or run "
            "`ollama serve`.\n"
            f"  - Model not pulled. Run `ollama pull {model_name}` "
            "(~5 GB download).\n"
            "  - Different tag on your machine. Pass --ollama_model "
            "yourtag to override."
        ) from e

    content = response.get("message", {}).get("content", "")
    classes = _parse_class_list(content)
    if not classes:
        raise RuntimeError(
            f"VLM returned no usable class names. Raw response was:\n"
            f"---\n{content}\n---"
        )
    return classes
