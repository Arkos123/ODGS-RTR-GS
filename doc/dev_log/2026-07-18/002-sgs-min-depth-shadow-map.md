---
created_at: "2026-07-18"
updated_at: "2026-07-18"
---

# SGS Min-Depth Shadow Map

## Summary

Equirectangular point-light shadow maps now use a min-depth geometry mode in
the SGS rasterizer. The regular SGS geometry depth remains alpha-weighted by
default; only the point-light shadow-map path opts into min-depth.

## Changes

- Added `depth_mode` to `GaussianRasterizationSettings`.
- Routed `depth_mode` through the SGS Python binding, C++ wrapper, rasterizer
  interface, and CUDA geometry pass.
- Added `depth_mode=1` to `pbr.point_light_shadow.get_depth_equirect()` so the
  equirect shadow map stores the nearest contributing Gaussian center depth.
- Kept `depth_mode=0` as the default for existing training and visualization
  behavior.
- Added viewer shortcut `Z` to switch point-light shadow depth between
  nearest/argmax-style depth and alpha-weighted depth for quick debugging.
- Added viewer shortcut `Y` to save the point-light shadow-map depth currently
  used by rendering, including metadata for backend, light position, and depth
  mode.
- Included the shadow depth mode in the equirect viewer cache key so switching
  `Z` forces a fresh render instead of reusing the previous panorama.
- Added mouse-wheel point-light intensity adjustment in the viewer. Hold
  `Shift` while scrolling to keep the previous orbit zoom behavior.
- Tuned the interactive point-light defaults used by the viewer during shadow
  debugging: world-axis light movement, `intensity=50.0`, and a `0.25` shadow
  threshold.

## Validation

- `python3 -m compileall pbr/point_light_shadow.py viewer_clean.py`
- `python3 -m compileall submodules/spherical-gaussian-splatting/submodules/spherical-gaussian-rasterization/spherical_gaussian_rasterization/__init__.py`
- Rebuilt `spherical_gaussian_rasterization` in `odgs-rtr` with
  `pip install --no-build-isolation -e .`.
- CUDA smoke test with two same-direction Gaussians:
  alpha-weighted mean depth was `2.054574`, while min-depth mode produced
  `2.000000` with unchanged alpha.
