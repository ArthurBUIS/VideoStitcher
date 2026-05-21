"""
Depth-aware static-FG selector (runtime, per-detection).

When --fg_model is yoloe, the main pipeline asks YOLOE for the
combined ALWAYS_KEEP + FOREGROUND_ONLY vocabulary, then at every FG
recompute tick (--fg_recompute_seconds, default 10s) it runs Depth
Anything V2 on each source frame and DROPS each detection whose
class is in FOREGROUND_ONLY and whose bbox-median depth is at or
below FG_DEPTH_THRESHOLD.

Per-detection (not per-class) is the key: two chairs from the same
YOLOE call get independent verdicts. A foreground chair gets
included; the background chair next to the wall gets dropped.

Tier semantics:

  ALWAYS_KEEP         classes that go into the YOLOE vocab AND are
                      always preserved at runtime (TV, monitor,
                      picture frame, ...). Things a seam should
                      never cross regardless of depth.
  FOREGROUND_ONLY     classes that go into the YOLOE vocab but are
                      filtered at runtime by per-bbox depth.
                      Foreground instances kept, background dropped.

The "non-important" tier is implicit: any class not in either list
is simply not asked of YOLOE.

Edit the two lists below to tune. Edit FG_DEPTH_THRESHOLD (or pass
--static_fg_depth_threshold on the CLI) to shift the foreground
cut-off.

Cost: per FG recompute (~once every 10s), 2x Depth Anything V2
calls (one per source frame). ~600 ms total at 1440p on a T1000,
amortized to <1% pipeline overhead.
"""


# ---------------------------------------------------------------------------
# Editable tier lists
# ---------------------------------------------------------------------------
#
# Class names are YOLOE text prompts (any reasonable noun phrase).
# Multi-word phrases are fine (YOLOE matches via Mobile-CLIP).

ALWAYS_KEEP = [
    "tv",
    "monitor",
    "computer",
    "laptop",
    "screen",
    "picture frame",
    "poster",
]

FOREGROUND_ONLY = [
    "chair",
    "armchair",
    "office chair",
    "stool",
    "couch",
    "sofa",
    "desk",
    "table",
    "coffee table",
    "dining table",
    "side table",
    "bookshelf",
    "cabinet",
    "plant",
    "potted plant",
    "houseplant",
    "floor lamp",
]

# Normalized-depth threshold for the FOREGROUND_ONLY tier. Depth is
# in [0, 1] where 1.0 = closest to the camera, 0.0 = farthest (see
# stitcher.depth_estimation.normalize_depth). 0.4 lets through
# clearly-foreground instances and drops mid/background ones.
FG_DEPTH_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# Helpers used by the pipeline at startup
# ---------------------------------------------------------------------------


def get_combined_vocab(always_keep=None, foreground_only=None):
    """
    YOLOE text-class list to use as fg_segmenter's vocabulary. Order:
    ALWAYS_KEEP first, then FOREGROUND_ONLY entries that aren't
    already in ALWAYS_KEEP. The resulting list is what YOLOE looks
    for in every frame.
    """
    if always_keep is None:
        always_keep = ALWAYS_KEEP
    if foreground_only is None:
        foreground_only = FOREGROUND_ONLY
    fg_only_dedup = [c for c in foreground_only if c not in always_keep]
    return list(always_keep) + fg_only_dedup


def get_fg_only_indices(vocab, foreground_only=None):
    """
    Class IDs within `vocab` (= the list passed to YOLOE) that
    correspond to FOREGROUND_ONLY entries -- the indices the runtime
    depth filter checks. Pass the same list you fed into YOLOE.
    """
    if foreground_only is None:
        foreground_only = FOREGROUND_ONLY
    fg_only_set = set(foreground_only)
    return [i for i, name in enumerate(vocab) if name in fg_only_set]
