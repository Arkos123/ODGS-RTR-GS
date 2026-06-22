"""
Render an equirectangular panorama from a perspective-mode checkpoint by
rendering 6 cubemap faces at a given world position and stitching them.

Usage (via render_checkpoint.py):
    python render_checkpoint.py \\
        -s <dataset_path> -m <output_path> -c <checkpoint.pth> \\
        -t render_ref \\
        --render_equirect \\
        --cubemap_position 0 0 0 \\
        --face_res 512
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from scene.cameras import Camera
from utils.graphics_utils import cubemap_to_equirect


# ── nvdiffrast cubemap face order ────────────────────────────────────
FACE_DIRECTIONS = [
    np.array([ 1.0,  0.0,  0.0]),   # 0: +X
    np.array([-1.0,  0.0,  0.0]),   # 1: -X
    np.array([ 0.0,  1.0,  0.0]),   # 2: +Y
    np.array([ 0.0, -1.0,  0.0]),   # 3: -Y
    np.array([ 0.0,  0.0,  1.0]),   # 4: +Z
    np.array([ 0.0,  0.0, -1.0]),   # 5: -Z
]

# Up vector per face (avoid degeneracy for +Y / -Y)
FACE_UP = [
    np.array([0.0, 1.0,  0.0]),     # +X
    np.array([0.0, 1.0,  0.0]),     # -X
    np.array([0.0, 0.0,  1.0]),     # +Y  (default up parallel to forward → use +Z)
    np.array([0.0, 0.0, -1.0]),     # -Y
    np.array([0.0, 1.0,  0.0]),     # +Z
    np.array([0.0, 1.0,  0.0]),     # -Z
]

FACE_NAMES = ["posx", "negx", "posy", "negy", "posz", "negz"]


# ── helpers ──────────────────────────────────────────────────────────

def _c2w_rotation(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Build 3×3 camera-to-world rotation matrix (column-major).

    Uses the standard GLM look-at convention:
       right = forward × up_world
       true_up = right × forward
       C2W_rot = [right | true_up | -forward]
    which guarantees det = +1 (pure rotation, no reflection).
    """
    forward = forward / np.linalg.norm(forward)
    up = up / np.linalg.norm(up)
    right = np.cross(forward, up)            # F × U = right (RH)
    n = np.linalg.norm(right)
    if n < 1e-8:                             # forward ∥ up → fallback
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        n = np.linalg.norm(right)
        if n < 1e-8:
            right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
            n = np.linalg.norm(right)
    right = right / n
    true_up = np.cross(right, forward)       # orthogonalise
    c2w = np.column_stack([right, true_up, -forward])
    return c2w


def _make_canonical_rays(H: int, W: int, fov: float) -> torch.Tensor:
    """Build canonical camera-space ray directions for a pinhole camera.

    These have the same format as ``scene.Scene.get_canonical_rays``
    but accept arbitrary resolution & FOV (needed per cubemap face).
    Returns [H*W, 3] on CUDA.
    """
    cen_x = W / 2
    cen_y = H / 2
    tan_hf = math.tan(fov * 0.5)
    focal_x = W / (2.0 * tan_hf)
    focal_y = H / (2.0 * tan_hf)

    x, y = torch.meshgrid(
        torch.arange(W, device="cuda"),
        torch.arange(H, device="cuda"),
        indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()
    dirs = F.pad(
        torch.stack([(x - cen_x + 0.5) / focal_x,
                      (y - cen_y + 0.5) / focal_y], dim=-1),
        (0, 1), value=1.0,
    )
    return dirs  # [H*W, 3]


def make_cubemap_camera(position: np.ndarray,
                        forward: np.ndarray,
                        up: np.ndarray,
                        face_res: int,
                        fov_deg: float = 90.0,
                        uid: int = 0) -> Camera:
    """Create a Camera looking in *forward* direction from *position*.

    The returned Camera can be fed to the perspective render function.
    FOV is 90° by default (standard for cubemap faces).

    Coordinate notes:
      - ``position``, ``forward``, ``up`` are all in **COLMAP world space**
        (+X right, +Y down, +Z forward).
      - Internally, ``_c2w_rotation`` builds C2W = [right | up | -forward],
        so the camera's +Z (forward direction in camera space) maps to
        *-forward* in world.  For example, passing forward=(0, +1, 0)
        creates a camera that looks *up* (in COLMAP convention, -Y is up).
      - This ensures the rendered face images are in nvdiffrast/OpenGL
        cubemap convention: face +Y = sky, face -Y = ground.
    """
    # ═══════════════════════════════════════════════════════════════
    #  WARNING — fragile Camera construction for cubemap faces.
    # ═══════════════════════════════════════════════════════════════
    #
    #  Camera R = c2w_rot (NOT w2c_rot), because getWorld2View(R,T)
    #  transposes it:  W2C = [[R.T, T], [0,0,0,1]] = [[c2w.T, T],...]
    #
    #  Then world_view_transform = W2C.transpose(0,1), read column-
    #  major by the CUDA rasterizer → effective rotation = c2w (not
    #  c2w.T).  The net transform becomes:
    #
    #      result = c2w @ p + T    with T = -c2w.T @ pos
    #
    #  For forward = ±X or ±Z the matrix c2w happens to be symmetric
    #  (c2w == c2w.T), so the error in the rotation direction is
    #  invisible.  For forward = ±Y the matrix is asymmetric → the
    #  rendered +Y and -Y face images are rotated 180° relative to
    #  what nvdiffrast's cubemap convention expects.
    #
    #  Therefore render_equirect_from_position MUST apply torch.flip
    #  on dims=[1,2] (180° rotation) to faces 2 (+Y) and 3 (-Y) after
    #  rendering before storing them into the cubemap tensor.
    #
    #  See render_equirect_from_position() for that compensation.
    # ═══════════════════════════════════════════════════════════════
    c2w_rot = _c2w_rotation(forward, up)
    R = c2w_rot
    T = -c2w_rot.T @ np.array(position)

    fov = math.radians(fov_deg)
    return Camera(
        colmap_id=uid,
        R=R,
        T=T,
        FoVx=fov,
        FoVy=fov,
        fx=None,
        fy=None,
        cx=None,
        cy=None,
        image_name=f"cubemap_{FACE_NAMES[uid]}",
        uid=uid,
        height=face_res,
        width=face_res,
        render_only=True,
    )


# ── main public function ─────────────────────────────────────────────

def render_equirect_from_position(
    position=None,
    gaussians=None,
    pipe=None,
    render_fn=None,
    dict_params=None,
    face_res: int = 512,
    eq_width: int = 2048,
    eq_height: int = 1024,
    viewpoint=None,
    yaw: float = 0.0,
):
    """Render an equirectangular panorama from a 3D position.

    Works in *perspective* (non-equirect) mode only.

    Exactly one of ``position`` or ``viewpoint`` must be given.

    Coordinate conventions:
      - ``position`` is in **COLMAP world space** (+X right, +Y down,
        +Z forward).  When ``viewpoint`` is used its ``camera_center``
        provides the position.
      - Each cubemap face is rendered via ``make_cubemap_camera``, which
        follows the nvdiffrast/OpenGL cubemap convention internally:
        face +Y (index 2) = **sky** (up), face -Y (index 3) = **ground**
        (down), face +Z (index 4) = **forward**.
      - The returned ``cubemaps`` dict contains tensors in **nvdiffrast**
        order (+X(0), -X(1), +Y(2), -Y(3), +Z(4), -Z(5)), and the
        stitched equirects have **sky at the top, ground at the bottom**.
      - See ``utils.graphics_utils.cubemap_to_equirect`` for the stitching
        convention and ``make_cubemap_camera`` for the per-face transform.

    All channels from the render output that share the same spatial
    shape (C, H, W) as ``"render"`` are automatically stitched to
    equirect.  This includes direct keys (render, depth, opacity,
    normal, pseudo_normal, pbr) and ``vis_dict`` keys (base_color,
    roughness, metallic, …).

    Args:
        position: length-3 array/list (camera centre in COLMAP world space).
            Mutually exclusive with ``viewpoint``.
        gaussians: GaussianModel (already on GPU).
        pipe: PipelineParams.
        render_fn: One of ``render`` or ``render_fast`` from
            ``gaussian_renderer`` (perspective renderers).
        dict_params: kwargs forwarded to ``render_fn`` (refmap, cubemap,
            brdf_lut, transfer_net, canonical_rays, …).
        face_res: Resolution of each cubemap face (H == W).
        eq_width, eq_height: Output equirectangular panorama resolution.
        viewpoint: A Camera whose ``camera_center`` is used as the
            position.  Mutually exclusive with ``position``.
        yaw: Yaw rotation in **degrees**.  Positive values rotate the
            equirect centre rightward (e.g. ``yaw=90`` puts +X at
            centre).  Default 0 (+Z at centre).

    Returns:
        A dict with:
          - ``"equirects"``: ``{channel_name: [C, eq_h, eq_w]}``
          - ``"cubemaps"``:  ``{channel_name: [6, H, W, C]}``
          - ``"face_results"``: list of 6 raw render_pkg dicts
    """
    from gaussian_renderer.render import render as render_fn_default
    if render_fn is None:
        render_fn = render_fn_default

    if viewpoint is not None:
        pos = viewpoint.camera_center.detach().cpu().numpy()
    elif position is not None:
        pos = np.asarray(position, dtype=np.float64)
    else:
        raise ValueError("Exactly one of position or viewpoint must be given")

    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    fov_rad = math.radians(90.0)

    face_results = []

    for i in range(6):
        cam = make_cubemap_camera(
            pos, FACE_DIRECTIONS[i], FACE_UP[i],
            face_res=face_res, uid=i,
        )

        face_rays = _make_canonical_rays(face_res, face_res, fov_rad)
        face_dict = dict(dict_params)
        face_dict["canonical_rays"] = face_rays

        pkg = render_fn(
            cam, gaussians, pipe, bg,
            is_training=False, dict_params=face_dict,
        )
        face_results.append(pkg)

    # ── Auto-detect channels ──
    # Non-training renders produce a vis_dict with gamma-corrected,
    # background-blended outputs ready for direct saving.  Use it as the
    # primary source; supplement with pkg-level channels (render, pbr)
    # that only exist outside vis_dict.
    ref_spatial = face_results[0]["render"].shape[-2:]  # (H, W)
    channels = {}

    def _maybe_add(key, val, prefix):
        if isinstance(val, torch.Tensor) and val.ndim == 3 \
                and val.shape[-2:] == ref_spatial:
            channels[f"{prefix}:{key}"] = key

    for key, val in face_results[0].get("vis_dict", {}).items():
        _maybe_add(key, val, "vis_dict")
    # Supplement with pkg-level channels not present in vis_dict.
    for key in face_results[0]:
        if key != "vis_dict" and key not in face_results[0].get("vis_dict", {}):
            _maybe_add(key, face_results[0][key], "pkg")

    # ── Build cubemap cubes and stitch to equirect for every channel ──
    equirects = {}
    cubemaps = {}

    for label, chan_key in channels.items():
        cube_list = []
        for i, face_pkg in enumerate(face_results):
            # vis_dict channels are already post-processed (gamma, bg-blend,
            # *0.5+0.5); pkg channels are raw/linear.
            if label.startswith("vis_dict:"):
                src = face_pkg.get("vis_dict", {}).get(chan_key)
            else:
                src = face_pkg.get(chan_key)
                if src is None:
                    src = face_pkg.get("vis_dict", {}).get(chan_key)
            if src is None:
                break
            d = src.clone()
            # +Y/-Y faces (indices 2, 3) are rendered 180° rotated relative
            # to nvdiffrast's cubemap convention due to an asymmetric c2w
            # matrix in make_cubemap_camera().  See that function's
            # docstring for the full explanation.
            if i in (2, 3):
                d = torch.flip(d, dims=[1, 2])
            cube_list.append(d.permute(1, 2, 0).unsqueeze(0))

        if len(cube_list) != 6:
            continue
        cube = torch.cat(cube_list, dim=0)  # [6, H, W, C]
        cubemaps[label] = cube

        eq = cubemap_to_equirect(cube, eq_width, eq_height)
        if yaw != 0.0:
            shift = int(round(yaw / 360.0 * eq_width))
            eq = torch.roll(eq, shifts=-shift, dims=-1)
        equirects[label] = eq

    return {
        "equirects": equirects,
        "cubemaps": cubemaps,
        "face_results": face_results,
    }
