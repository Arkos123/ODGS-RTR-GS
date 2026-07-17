---
created_at: "2026-07-17"
updated_at: "2026-07-17"
---

# Viewer Point Light and Equirect Alignment

## Summary

Updated `viewer_clean.py` to improve the real-time relighting viewer:

- Added an on-screen point-light indicator rendered as a glowing sphere at the projected light position.
- Added a three-state `N` display toggle:
  - `PERSPECTIVE`
  - `EQUIRECT CROP`
  - `EQUIRECT PANORAMA`
- Fixed viewer camera conventions so perspective movement, point-light overlay, and equirect crop use consistent axes.
- Added `AGENTS.md` as a concise Codex contributor guide.

## Coordinate Fixes

The perspective viewer now treats camera canonical rays as:

```text
[x_right, y_down, +z_forward]
```

`CameraController.build_camera()` builds `c2w` columns as:

```text
[screen_right, screen_down, forward]
```

This aligns the interactive viewer with the renderer's canonical ray convention and fixes inverted forward/up movement.

For equirect crop display, the viewer no longer rebuilds a separate camera basis. It reuses the same perspective camera path:

```python
persp_camera, canonical_rays = cam_ctrl.build_camera()
```

Then `_equirect_to_perspective()` derives world-space sample directions from `canonical_rays + c2w`. This keeps equirect crop behavior aligned with normal perspective rendering and avoids side-view pitch turning into roll.

## Notes

- Raw equirect panorama display remains a direct view of the rendered ERP image.
- The point-light indicator is disabled in raw panorama mode because it is a perspective-screen overlay.
- Validation performed: `python3 -m py_compile viewer_clean.py`.
