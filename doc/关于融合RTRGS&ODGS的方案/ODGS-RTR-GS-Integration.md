# ODGS ↔ RTR-GS Integration

**Goal**: Use ODGS to reconstruct 3DGS geometry from equirectangular (360°) image sequences, then feed the reconstruction into RTR-GS for inverse rendering (BRDF/lighting decomposition).

**Constraint**: All input data is equirectangular panoramas — there are no perspective NVS images available as ground truth.

---

## Pipeline Overview

```
                   ┌──────────────────────────────────────┐
                   │  ODGS Training (equirect)            │
 Input: panorama   │  - reconstructs 3DGS geometry        │
 sequence ────────►│  - outputs .ply (6 attrs) + chkpnt   │
                   └──────────┬───────────────────────────┘
                              │
                              ▼
                   ┌──────────────────────────────────────┐
                   │  PLY Conversion +                    │
                   │  RTR-GS Attribute Init               │
                   └──────────┬───────────────────────────┘
                              │
                              ▼
                   ┌──────────────────────────────────────┐
                   │  RTR-GS Equirect Inverse Rendering   │
                   │  - uses ODGS CUDA rasterizer         │
 Input: panorama   │  - multi-pass forward rendering      │
 sequence ────────►│  - PRT + reflection + PBR            │
                   │  - latitude-weighted losses          │
                   └──────────────────────────────────────┘
```

---

## Architecture Analysis: Why This Is Feasible

RTR-GS's rendering pipeline has a **clean 3-layer separation** that makes swapping the rasterizer practical:

```
[Layer 1] Python: per-Gaussian color computation
    - PRT colors (diffuse + specular)
    - Reflection colors (forward shading)
    - Assembly of per-Gaussian attributes
          │
          ▼
[Layer 2] CUDA: splatting + alpha blending (PROJECTION-AWARE)
    - Tile-based sorting & culling
    - Per-pixel alpha compositing:
      • colors_precomp (RGB)      → rendered_image
      • features (10~18D tensors) → rendered_feature (if supported)
          │
          ▼
[Layer 3] Python: pixel-level deferred shading (PROJECTION-AGNOSTIC)
    - Normal map extraction
    - Reflection rendering
    - PBR shading
```

**Key insight**: Only Layer 2 (the CUDA rasterizer) knows about the projection model. Layers 1 and 3 are projection-agnostic — they work on per-Gaussian or per-pixel data regardless of whether the projection is perspective or equirectangular.

---

## The Main Challenge: Feature Blending

RTR-GS's deferred PBR relies on the CUDA rasterizer's `features` tensor mechanism — it packs per-Gaussian attributes (depth, normal, BRDF params) into a multi-channel tensor, and the CUDA kernel alpha-blends each channel independently.

**ODGS's CUDA rasterizer does NOT support this `features` mechanism**. It only handles `colors_precomp` (RGB) and `sh` (SH coefficients).

### Solution: Multi-Pass Forward Rendering

Instead of modifying the ODGS CUDA kernel (significant effort), render each required channel as a separate forward pass:

| Pass | colors_precomp = | Output = | Purpose |
|------|-----------------|----------|---------|
| 1 | PRT radiance (RGB) | `radiance_map` | Base diffuse + specular color |
| 2 | Normal (encoded as RGB) | `normal_map` | Per-pixel surface normal |
| 3 | BRDF params (encoded as RGB) | `brdf_map` | base_color, roughness, metallic |
| 4 | Depth (float as RGB) | `depth_map` | Per-pixel depth |

All passes use the **same** ODGS CUDA rasterizer with the **same** camera parameters. The per-Gaussian data is identical (same positions, opacities, scales, rotations) — only `colors_precomp` changes.

After rendering, decode each pass back to its semantic meaning and run standard RTR-GS pixel-level PBR shading.

**Performance cost**: ODGS rasterizer is fast (~2ms per pass at 1024x512). 4 passes = ~8ms, acceptable.

---

## Work Breakdown

### Phase 1: PLY Bridge (required, low effort)

Write `scripts/odgs2rtrgs.py` that:
1. Loads ODGS .ply (6 attrs: xyz, SH, scale, rot, opacity)
2. Creates RTR-GS GaussianModel
3. Initializes RTR-GS extended attributes to defaults:
   - `diffuse_tint`, `specular_tint`, `ref_tint` = zeros
   - `ref_strength` = sigmoid⁻¹(0.01)
   - `ref_roughness` = sigmoid⁻¹(0.65)
   - `specular_feature` = zeros
   - `diffuse_transfer_dc/rest` = zeros
   - PBR: `base_color`, `roughness`, `metallic` = defaults
4. Saves as RTR-GS format .ply

Also fix the RTR-GS `load_ply` variable name bugs at [scene/gaussian_model.py](file:///home/huangpengyue/projects/RTR-GS/scene/gaussian_model.py#L718-L746).

### Phase 2: Equirectangular Camera Support (required, low effort)

1. Add equirectangular dataset reader to RTR-GS's [scene/dataset_readers.py](file:///home/huangpengyue/projects/RTR-GS/scene/dataset_readers.py) — reuse ODGS's OpenMVG parser
2. Camera class adapts naturally: ODGS uses same `Camera(R, T, FoVx, FoVy, ...)` as RTR-GS, with `FoVx = π` hardcoded

### Phase 3: Equirectangular Renderer (core work, medium effort)

Create `gaussian_renderer/render_equirect.py`:
1. Reuses RTR-GS PRT computation (projection-agnostic)
2. Replaces `diff_gaussian_rasterization` with `odgs_gaussian_rasterization`
3. Removes perspective-only params: `tanfovx/tanfovy/cx/cy/projmatrix`
4. Implements multi-pass rendering for deferred data
5. Adds latitude-weighted loss functions

### Phase 4: Training Pipeline (medium effort)

Modify `train.py` to support equirectangular mode:
1. Load ODGS .ply or checkpoint as initial geometry
2. Disable densification (already done = Stage 2 behavior)
3. Use equirectangular renderer
4. Use latitude-weighted L1 + SSIM loss

---

## Progress Checklist

- [ ] Phase 1: PLY conversion script + load_ply bugfix
- [ ] Phase 2: Equirectangular dataset reader + camera
- [ ] Phase 3: `render_equirect.py` (multi-pass forward rendering)
- [ ] Phase 3: Latitude-weighted losses
- [ ] Phase 4: Training script modifications
- [ ] End-to-end test

---

## Estimated Effort

| Phase | Files to modify | Lines of code | Difficulty |
|-------|----------------|:-------------:|:----------:|
| Phase 1 | 1 new + 1 modified | ~80 | Low |
| Phase 2 | 1 modified | ~100 | Low |
| Phase 3 | 1 new | ~300 | Medium |
| Phase 4 | 1 modified | ~100 | Medium |
| **Total** | **~5 files** | **~580** | |
