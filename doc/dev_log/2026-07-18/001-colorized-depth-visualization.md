---
created_at: "2026-07-18"
updated_at: "2026-07-18"
---

# Colorized Depth Visualization

## Summary

Depth outputs used for visual inspection now use a Turbo colormap instead of
plain 0-1 grayscale. The default mapping is logarithmic so nearby depth
differences are easier to read in both perspective and equirectangular views.

## Changes

- Added `utils.image_utils.colorize_depth()` with `curve="log"` by default and
  `curve="linear"` for linear visualization.
- Updated training, evaluation, checkpoint rendering, and relighting video
  export paths to colorize only depth outputs while leaving other single-channel
  material maps grayscale.
- Updated `viewer_clean.py` so the `Depth` view uses the same colorized depth
  path, including already-expanded 3-channel depth tensors from `vis_dict`.
- Added `C` as the previous-channel shortcut in `viewer_clean.py`, paired with
  existing `V` next-channel cycling.

## Validation

- `python3 -m compileall utils/image_utils.py train.py scripts/render_checkpoint.py render_and_eval.py eval_relighting_colmap.py`
- `python3 -m compileall viewer_clean.py`
- `conda run -n odgs-rtr` smoke tests for `colorize_depth()` raw,
  normalized, and 3-channel depth inputs.
- Rendered one equirectangular checkpoint view with
  `scripts/render_checkpoint.py -t render_ref_pbr_equirect` and verified the
  output depth PNG is RGB with non-identical channels.
