"""
Standalone CLI for auto-FG-v2 discovery.

Runs Steps 0+1+2+3 on frame 0 of a video and writes the result to a
JSON file (default: auto_fg_classes.json in the current dir). The
main video pipeline picks the file up via --yoloe_fg_classes auto.

Also prints the kept-class list to stdout (comma-separated, no
trailing punctuation) so it can still be piped or eyeballed without
reading the JSON.

Usage:
    python tools/suggest_fg_classes_v2.py --video videos/sf_left.mp4

    # Then in the same dir:
    python video_stitcher_seam_gpu.py \\
        --video_a videos/sf_left.mp4 --video_b ... --output out.mp4 \\
        --yoloe_fg_classes auto

Detailed logs (vocab, per-class decisions with reasons, raw VLM
response) go to stderr so stdout stays clean for piping.
"""

import argparse
import datetime as _dt
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from stitcher.auto_fg_v2 import (  # noqa: E402
    DEFAULT_AUTO_FG_JSON,
    read_frame_zero,
    suggest_fg_classes_v2,
)
from stitcher.vlm_judge import DEFAULT_OLLAMA_MODEL  # noqa: E402


def _records_to_json(records):
    """Flatten the depth-annotated records into JSON-friendly dicts."""
    out = []
    for r in records:
        out.append({
            "class": r["class"],
            "bbox": [float(v) for v in r["bbox"]],
            "conf": float(r["conf"]),
            "depth": float(r["depth"]),
        })
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Run the auto-FG-v2 discovery flow on frame 0 "
                    "of a video; write a JSON the main pipeline can "
                    "consume."
    )
    parser.add_argument("--video", required=True,
                        help="Path to a video; frame 0 is read.")
    parser.add_argument("--output_json", default=DEFAULT_AUTO_FG_JSON,
                        help=f"Where to write the discovery result. "
                             f"Default: {DEFAULT_AUTO_FG_JSON!r} (the "
                             f"main pipeline reads from the same path "
                             f"by default).")
    parser.add_argument("--yoloe_weights", default="yoloe-11s-seg.pt",
                        help="YOLOE weights path.")
    parser.add_argument("--device", default="cuda:0",
                        help="Torch device. Default: cuda:0.")
    parser.add_argument("--ollama_model", default=DEFAULT_OLLAMA_MODEL,
                        help="Ollama model tag used for BOTH the "
                             "inventory call (step 0) and the judge "
                             "call (step 3).")
    parser.add_argument("--min_confidence", type=float, default=0.0,
                        help="Drop YOLOE detections below this score.")
    parser.add_argument("--separator", default=", ",
                        help="String between class names on stdout. "
                             "Default: ', '. Pass ' ' for shell-"
                             "friendly piping into --yoloe_fg_classes.")
    args = parser.parse_args()

    print(f"[suggest_fg_classes_v2] reading frame 0 from {args.video}",
          file=sys.stderr)
    frame = read_frame_zero(args.video)
    H, W = frame.shape[:2]
    print(f"[suggest_fg_classes_v2] frame shape: H={H} W={W}",
          file=sys.stderr)
    print(f"[suggest_fg_classes_v2] running inventory + depth + VLM judge "
          f"(ollama: {args.ollama_model})...", file=sys.stderr)

    classes, vocab, records, decisions, raw = suggest_fg_classes_v2(
        frame,
        yoloe_weights=args.yoloe_weights,
        device=args.device,
        ollama_model=args.ollama_model,
        min_confidence=args.min_confidence,
        return_details=True,
    )

    # Detailed log -> stderr
    print(f"[suggest_fg_classes_v2] step 0 inventory vocab "
          f"({len(vocab)} phrases):", file=sys.stderr)
    for p in vocab:
        print(f"  - {p}", file=sys.stderr)
    print(f"[suggest_fg_classes_v2] step 1+2 detections "
          f"({len(records)}):", file=sys.stderr)
    for r in records:
        print(f"  d={r['depth']:.2f}  conf={r['conf']:.2f}  "
              f"{r['class']}", file=sys.stderr)
    print(f"[suggest_fg_classes_v2] step 3 per-class decisions "
          f"({len(decisions)}):", file=sys.stderr)
    for d in decisions:
        flag = "KEEP" if d["keep"] else "drop"
        print(f"  [{flag}] {d['name']:<22s} {d['reason']}",
              file=sys.stderr)
    print("[suggest_fg_classes_v2] step 3 raw VLM JSON:",
          file=sys.stderr)
    print("  " + raw.replace("\n", "\n  "), file=sys.stderr)
    print(f"[suggest_fg_classes_v2] selected ({len(classes)}): "
          f"{classes}", file=sys.stderr)

    # Persist for the main pipeline to consume.
    payload = {
        "kept_classes": classes,
        "source_video": args.video,
        "ollama_model": args.ollama_model,
        "created_at": _dt.datetime.now(_dt.timezone.utc)
                                  .isoformat(timespec="seconds"),
        "inventory_vocab": vocab,
        "detections": _records_to_json(records),
        "judge_decisions": decisions,
        "judge_raw_response": raw,
    }
    out_dir = os.path.dirname(os.path.abspath(args.output_json))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[suggest_fg_classes_v2] wrote {args.output_json}",
          file=sys.stderr)

    # Clean stdout for piping.
    sys.stdout.write(args.separator.join(classes))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
