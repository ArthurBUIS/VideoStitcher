"""
VLM judge: pick which detected classes the panorama seam should avoid.

Step C of the auto-FG-v2 discovery flow. Given:
  - one camera frame
  - the depth-annotated inventory from
    stitcher.depth_inventory.annotate_inventory_with_depth

ask a locally-hosted vision-language model (Qwen2.5-VL via Ollama by
default) to judge which classes in the inventory are visually
important enough that the stitching seam should route around them.

The VLM ranks by *importance*, not by foreground/background. A TV
mounted on the back wall is far in depth but visually critical; a
rug in the foreground is close in depth but a seam through it is
invisible. Depth goes into the prompt as a hint, but the VLM is the
final decider.

Output is JSON-schema-constrained per-item decisions:
    {"decisions": [{"name", "keep", "reason"}, ...]}
which forces the model to commit a boolean per class (with one
short sentence of justification). This pattern is known to work
better than free-form lists on sub-7B Qwen2.5-VL variants -- the
model has to think about each candidate twice instead of skimming.

The prompt itself is a POSITIVE WHITELIST (keep only items that
satisfy ALL of these rules) rather than an EXCLUDE-list-with-
worked-examples; the research showed text-only few-shots aren't
internalised by Qwen2.5-VL the way image-paired ones are.

Why single-image, single-message: Qwen2.5-VL on Ollama has been
unreliable with multi-image or composite inputs. One frame in one
message is the only call shape that consistently behaves.
"""

import cv2


DEFAULT_OLLAMA_MODEL = "qwen2.5vl:7b"


# Image is downscaled before being sent to the VLM so the call fits
# comfortably on a T1000 (8 GB). Long-side limit at 896 keeps the
# token count low without losing recognisable objects.
_VLM_MAX_DIM = 896


# The judge prompt. Edit this directly to tune the keep rules. The
# {class_summary} placeholder is filled per call from the depth-
# annotated inventory.
#
# Design notes:
#   - Positive whitelist (rules to KEEP), not a negation list.
#   - No worked examples -- text-only few-shots don't help
#     Qwen2.5-VL and the model tended to over-include.
#   - The schema (built per call below) constrains output to a
#     fixed JSON shape with one decision per class; the prompt only
#     needs to communicate the rules + the output contract.
JUDGE_PROMPT_TEMPLATE = """\
You are helping configure a real-time video stitching system. A
panorama seam will be drawn between two camera views of the same
room. The seam can cause visible distortion where it crosses an
object, so we need to route the seam AROUND items that would
visibly break if a seam cut through them.

Look at the image. For EACH class in the list below, decide
keep = true or keep = false.

Set keep = true if ANY ONE of the following clearly applies:
  1. The class contains a screen, an image, or text that would
     visibly misalign if a seam crossed it. Examples: TVs, monitors,
     framed art, posters, whiteboards, screens of any kind -> true
  2. The class is an object belonging to the foreground, so its depth
     is higher than 0.4 -> true
     

Set keep = false ONLY when the class clearly satisfies one of these:
  3. Floor or wall coverings whose pattern repeats and the eye
    doesn't track (carpets, rugs, mats) -> false
  4. Structural background that's clearly not an object (walls,
    ceiling, doors, windows) -> false
  5. Decorative background objects that is not a screen or a frame
     (lamps, plants, furniture) -> false
  6. Any other objects that belong to the background and was not
     concerned by the 5 rules above -> false

Default when uncertain: keep = true. Over-keeping is cheap (the
seam planner can route around extras). Over-dropping is expensive
(the seam can then cut through important objects).

Depth is normalized: 0.0 = far from the camera, 1.0 = close. Each
entry below lists how many instances of the class were detected
and their depth range.

Detected classes:
{class_summary}

For EACH class in the list above, emit one decision. The "rule"
field must be the integer (1, 2, 3, 4, 5, or 6) of the rule that
triggered for that class. Pick exactly ONE rule per class. Use
class names EXACTLY as they appear in the list.
"""


# Rule number -> (keep, human-readable reason). Edit this in tandem
# with the rule list in JUDGE_PROMPT_TEMPLATE above. Rules 1+2 are
# keeps; the rest are drops. The reason text is what downstream
# code (test_vlm_judge, suggest_fg_classes_v2) shows in its logs.
_RULE_TO_KEEP_AND_REASON = {
    1: (True,  "screen / image / text (rule 1)"),
    2: (True,  "foreground object, depth > 0.4 (rule 2)"),
    3: (False, "floor / wall covering (rule 3)"),
    4: (False, "structural background (rule 4)"),
    5: (False, "decorative background object (rule 5)"),
    6: (False, "other background (rule 6)"),
}


def _build_decision_schema(class_names):
    """
    Build a JSON schema that constrains the VLM to emit exactly one
    decision per class in `class_names`. The `name` field is locked
    to the inventory enum so the model can't hallucinate names; the
    `rule` field is locked to integers 1..6 (one of the rules in
    JUDGE_PROMPT_TEMPLATE). Rule -> keep/reason conversion happens
    in _parse_decisions via _RULE_TO_KEEP_AND_REASON.
    """
    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "enum": list(class_names),
                        },
                        "rule": {
                            "type": "integer",
                            "enum": sorted(_RULE_TO_KEEP_AND_REASON),
                        },
                    },
                    "required": ["name", "rule"],
                },
            },
        },
        "required": ["decisions"],
    }


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
    bullet list with instance count + depth range per class.

    Ordered by max depth descending (closest classes first) -- a
    soft prior nudging the model toward foreground-first reasoning.
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


def _parse_decisions(raw_text, allowed):
    """
    Parse Ollama's JSON-schema-constrained response into a list of
    decision dicts: [{"name", "keep", "reason", "rule"}, ...].

    Each VLM entry carries a "rule" integer; this function looks up
    (keep, reason) in _RULE_TO_KEEP_AND_REASON and emits the same
    {name, keep, reason} shape downstream code already consumes,
    plus a "rule" field for visibility.

    Filters out entries whose name isn't in `allowed` or whose rule
    isn't in the mapping (defence against schema-validation surprises).
    Preserves the model's emission order.
    """
    import json

    text = (raw_text or "").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"VLM did not return JSON despite the schema constraint. "
            f"Parse error: {e}\nRaw response:\n---\n{text}\n---"
        )

    decisions_raw = obj.get("decisions", [])
    if not isinstance(decisions_raw, list):
        raise RuntimeError(
            f"VLM response missing 'decisions' array. Got:\n"
            f"---\n{text}\n---"
        )

    allowed_set = set(allowed)
    out = []
    seen = set()
    for d in decisions_raw:
        if not isinstance(d, dict):
            continue
        name = d.get("name")
        rule = d.get("rule")
        if not isinstance(name, str) or not isinstance(rule, int):
            continue
        if name not in allowed_set or name in seen:
            continue
        mapping = _RULE_TO_KEEP_AND_REASON.get(rule)
        if mapping is None:
            continue
        keep, reason = mapping
        seen.add(name)
        out.append({
            "name": name,
            "keep": keep,
            "reason": reason,
            "rule": rule,
        })
    return out


def judge_inventory(frame_bgr, records,
                    model_name=DEFAULT_OLLAMA_MODEL,
                    prompt_template=JUDGE_PROMPT_TEMPLATE,
                    return_details=False):
    """
    Ask the VLM which classes in `records` the panorama seam should
    avoid cutting through.

    The call uses Ollama's `format=` JSON-schema constraint to force
    one decision per inventory class, with a boolean `keep` and a
    one-sentence `reason`. The returned class list contains every
    class where keep == true.

    Args:
        frame_bgr: HxWx3 uint8 BGR image -- the SINGLE frame the VLM
            looks at (typically the left camera). Multi-image and
            composite inputs have been unreliable, so the call stays
            single-frame.
        records: list of dicts as returned by
            stitcher.depth_inventory.annotate_inventory_with_depth.
        model_name: Ollama tag of a Qwen2.5-VL model. Default
            qwen2.5vl:3b. Try qwen2.5vl:7b if you have VRAM headroom.
        prompt_template: edit JUDGE_PROMPT_TEMPLATE at module scope
            to tune what the VLM keeps, or pass an override here.
            Must contain a {class_summary} placeholder.
        return_details: if True, also return (decisions, raw_text)
            where decisions is the full per-item list (kept + dropped
            + reasons) and raw_text is the VLM's JSON string.

    Returns:
        list of class names to protect.
        If return_details is True: (classes, decisions, raw_text).

    Raises RuntimeError with an actionable hint when the `ollama`
    package is missing, the daemon isn't running, the model isn't
    pulled, the VLM returned malformed JSON, or no class was kept.
    """
    if not records:
        if return_details:
            return [], [], ""
        return []

    try:
        import ollama
    except ImportError as e:
        raise RuntimeError(
            "VLM judge requires the `ollama` Python package. Install with:\n"
            "    pip install ollama"
        ) from e

    class_summary = _summarize_inventory(records)
    # Debug: log the exact class_summary the VLM will see in the prompt.
    import sys as _sys
    print("[vlm_judge] class_summary injected into prompt:",
          file=_sys.stderr)
    print(class_summary, file=_sys.stderr)
    prompt = prompt_template.format(class_summary=class_summary)

    small = _downscale_long_side(frame_bgr, _VLM_MAX_DIM)
    img_png = _encode_png_bytes(small)

    allowed = sorted({r["class"] for r in records})
    schema = _build_decision_schema(allowed)

    try:
        response = ollama.chat(
            model=model_name,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [img_png],
            }],
            format=schema,
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
            "to override.\n"
            "  - ollama Python package older than 0.4 (format= support). "
            "Upgrade with: pip install -U ollama"
        ) from e

    raw = response.get("message", {}).get("content", "")
    decisions = _parse_decisions(raw, allowed)
    classes = [d["name"] for d in decisions if d["keep"]]
    if not classes:
        raise RuntimeError(
            "VLM kept no classes. Per-item decisions:\n"
            + "\n".join(f"  {d['name']}: keep={d['keep']} reason={d['reason']!r}"
                        for d in decisions)
            + f"\nRaw response:\n---\n{raw}\n---"
        )
    if return_details:
        return classes, decisions, raw
    return classes
