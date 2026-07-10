# tools

## Purpose
Standalone developer utilities that exercise parts of the `stitcher`
package outside the full pipeline. Currently a single validator for the
depth-aware static-foreground selector, used to tune the vocab tiers and
the depth threshold against a real frame before committing them.

## Contents
| File | Responsibility | Key symbols |
|------|----------------|-------------|
| `test_static_fg.py` | Simulate one FG-recompute tick: run YOLOE on the combined `ALWAYS_KEEP`+`FOREGROUND_ONLY` vocab over frame 0 of a video, run Depth Anything V2, print a per-detection keep/drop verdict, and save an annotated PNG (kept boxes coloured, dropped ones dim grey). | `main`, `_run_yoloe_boxes`, `_draw_annotated`, `_grab_frame_zero` |

## Data flow
Input: `--video PATH` (frame 0 is read) + `--yoloe_weights`, `--device`,
`--min_confidence`, `--depth_threshold`. →
`static_fg.get_combined_vocab` / `get_fg_only_indices` build the vocab and
the fg-only index set → `segmentation.PersonSegmenter(use_yoloe=True)` +
`predict_classes_boxes` return per-class bboxes → if any FOREGROUND_ONLY
class fired, `depth_estimation.estimate_depth` + `normalize_depth` →
each detection scored with `bbox_median_depth` and compared to the
threshold (mirrors `segmentation._depth_filter_keep_mask`). Output: printed
verdict table + `<video>_static_fg.png`.

## Dependencies
- **Internal:** `stitcher.static_fg`, `stitcher.depth_estimation`,
  `stitcher.segmentation`. Adds the repo root to `sys.path` so it runs from
  anywhere. It re-implements the keep/drop rule rather than calling the
  pipeline helper — keep it in sync with `_depth_filter_keep_mask`.
- **External:** `cv2`, `ultralytics` (YOLOE), `transformers`+`PIL` (depth),
  `torch` (GPU + memory release). CUDA GPU assumed by the default
  `--device cuda:0`.

## Key decisions & gotchas
- **Frame 0 only.** It does not stitch or read a second camera; it inspects
  a single frame to preview which objects the runtime filter would keep.
- Releases YOLOE and the depth model (`gc` + `torch.cuda.empty_cache` /
  `release_depth`) between stages to keep GPU memory low.
- The verdict logic is duplicated from the pipeline; a change to
  `_depth_filter_keep_mask` semantics must be reflected here too.

## Entry points
CLI: `python tools/test_static_fg.py --video videos/sf_left.mp4`.
Not imported by anything else.

## Pointers
- Parent / index: [../progress.md](../progress.md)
- Depends on: [../stitcher/progress.md](../stitcher/progress.md)
  (`static_fg`, `depth_estimation`, `segmentation`)
