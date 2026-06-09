"""
Render visualization images from a saved checkpoint.

Usage:
    # Equirect model (SGS/ODGS): 已完成全景训练的checkpoint
    python render_checkpoint.py \\
        -s <dataset_path> \\
        -m <output_path> \\
        -c <output_path>/checkpoint/chkpnt30000.pth \\
        -t render_ref_equirect \\
        --occlusion_path <output_path>/checkpoint/occlusion_volumes.pth \\
        --num_views 4

    # PBR equirect (Stage2):
    python render_checkpoint.py \\
        -s <dataset_path> \\
        -m <output_path> \\
        -c <output_path>/checkpoint/chkpnt40000.pth \\
        -t render_ref_pbr_equirect \\
        --occlusion_path <output_path>/checkpoint/occlusion_volumes.pth \\
        --num_views 4

Output:
    <model_path>/vis/checkpoint_vis/
        view_<name>/
            render.png       # 渲染结果
            gt.png           # 真值
            depth.png        # 深度图
            normal.png       # 法线图
            pseudo_normal.png # 伪法线图
            opacity.png      # 透明度
            pbr.png          # PBR结果（仅PBR模式）
"""
import os
import torch
import torchvision
import argparse
from scene import Scene, GaussianModel
from gaussian_renderer import render_fn_dict
from pbr import CubemapLight, get_brdf_lut
from scene.transfer_mlp import TransferMLP
from utils.general_utils import safe_state
from utils.graphics_utils import latlong_to_cubemap_equirect


def build_pipe():
    """Create PipelineParams with default values."""
    from argparse import ArgumentParser
    from arguments import PipelineParams
    p = ArgumentParser(add_help=False)
    pipe = PipelineParams(p)
    return pipe


def build_dataset(source_path, model_path, sh_degree=3):
    """Create dataset config via ModelParams with argparse defaults."""
    from argparse import ArgumentParser
    from arguments import ModelParams
    parser = ArgumentParser(add_help=False)
    mp = ModelParams(parser)
    ns = parser.parse_args([
        "--source_path", source_path,
        "--model_path", model_path,
        "--eval",
    ])
    ns.sh_degree = sh_degree
    return mp.extract(ns)


def load_component(checkpoint_dir, name, checkpoint_name, cls, **kwargs):
    """Load a checkpointed component (e.g. transfer_net, refmap, cubemap)."""
    ckpt_path = os.path.join(checkpoint_dir, f"{name}_{checkpoint_name}")
    has_ckpt = os.path.exists(ckpt_path)
    if not has_ckpt:
        print(f"  Warning: {name} checkpoint not found at {ckpt_path}, using fresh init")

    is_plain = cls.__name__ == "TransferMLP"  # not an nn.Module, no .cuda()/.eval()
    component = cls(**kwargs)
    if not is_plain:
        component = component.cuda()

    if has_ckpt:
        component.create_from_ckpt(ckpt_path, restore_optimizer=False)

    if not is_plain:
        component.eval()
    return component


def main():
    parser = argparse.ArgumentParser(description="Render visualization from a saved checkpoint")
    parser.add_argument("-s", "--source_path", required=True,
                        help="Dataset source path (images, cameras)")
    parser.add_argument("-m", "--model_path", required=True,
                        help="Model output directory (vis/ will be written here)")
    parser.add_argument("-c", "--checkpoint", required=True,
                        help="Path to Gaussian model checkpoint .pth file")
    parser.add_argument("-t", "--type", default="render_ref_equirect",
                        choices=list(render_fn_dict.keys()),
                        help="Render type")
    parser.add_argument("--occlusion_path", default=None,
                        help="Path to occlusion volumes .pth (needed for PBR mode)")
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--num_views", type=int, default=4,
                        help="Number of test views to render")
    parser.add_argument("--diffuse_iteration", type=int, default=3000,
                        help="Iterations before using full PRT (only_diffuse threshold)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    safe_state(args.quiet)
    os.makedirs(args.model_path, exist_ok=True)

    is_pbr = args.type in ('render_ref_pbr', 'render_ref_fast', 'render_ref_pbr_equirect')
    is_equirect = args.type in ('render_ref_equirect', 'render_ref_pbr_equirect')
    use_ref_map = args.type in ('render_ref', 'render_ref_pbr', 'render_ref_fast',
                                'render_ref_equirect', 'render_ref_pbr_equirect')

    checkpoint_dir = os.path.dirname(args.checkpoint)
    checkpoint_name = os.path.basename(args.checkpoint)

    # Build pipe with defaults, then override for our render type
    pipe = build_pipe()
    pipe.compute_with_prt = True
    pipe.forward_shading = True
    pipe.ref_map = use_ref_map
    pipe.diffuse_iteration = args.diffuse_iteration
    pipe.metallic = is_pbr
    pipe.equirect = is_equirect

    print(f"Render type: {args.type}")
    print(f"  PBR: {is_pbr}, Equirect: {is_equirect}, RefMap: {use_ref_map}")
    print(f"  Source: {args.source_path}")
    print(f"  Model:  {args.model_path}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Occlusion: {args.occlusion_path}")

    # ---- Setup Gaussians and Scene ----
    dataset = build_dataset(args.source_path, args.model_path, args.sh_degree)
    gaussians = GaussianModel(dataset.sh_degree, render_type=args.type)
    scene = Scene(dataset, gaussians)

    print(f"Loading checkpoint from {args.checkpoint} ...")
    gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=False)
    gaussians.active_sh_degree = gaussians.max_sh_degree
    print(f"  Loaded {gaussians.get_xyz.shape[0]} Gaussians")

    # ---- PBR components ----
    pbr_kwargs = {}

    # BRDF LUT (needed for reflection rendering)
    brdf_lut = get_brdf_lut().cuda()
    pbr_kwargs["brdf_lut"] = brdf_lut

    # Transfer MLP (for PRT rendering)
    if pipe.compute_with_prt:
        transfer_net = load_component(
            checkpoint_dir, "transfer_net", checkpoint_name,
            TransferMLP, sh_degree=gaussians.max_sh_degree,
            features_n=gaussians.n_featres)
        pbr_kwargs["transfer_net"] = transfer_net

    # Canonical rays (for perspective pinhole mode, not needed for equirect)
    if not is_equirect:
        pbr_kwargs["canonical_rays"] = scene.get_canonical_rays()

    # Reflection map
    if use_ref_map:
        refmap = load_component(checkpoint_dir, "refmap", checkpoint_name,
                                CubemapLight, base_res=128)
        refmap.build_mips()
        pbr_kwargs["refmap"] = refmap

    # Environment cubemap + occlusion (PBR mode only)
    if is_pbr:
        cubemap = load_component(checkpoint_dir, "cubemap", checkpoint_name,
                                 CubemapLight, base_res=128)
        cubemap.build_mips()
        pbr_kwargs["cubemap"] = cubemap

        if args.occlusion_path is not None:
            print(f"  Loading occlusion volumes from {args.occlusion_path}")
            occlusion_volumes = torch.load(args.occlusion_path)
            bound = occlusion_volumes["bound"]
            aabb = torch.tensor([-bound, -bound, -bound, bound, bound, bound]).cuda()
            pbr_kwargs["occlusion_volumes"] = occlusion_volumes
            pbr_kwargs["aabb"] = aabb

    # ---- Render ----
    render_fn = render_fn_dict[args.type]
    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    test_cameras = scene.getTestCameras()
    if not test_cameras:
        print("No test cameras found, falling back to training cameras")
        test_cameras = scene.getTrainCameras()

    n_views = min(args.num_views, len(test_cameras))
    vis_dir = os.path.join(args.model_path, "vis", "checkpoint_vis")
    print(f"\nRendering {n_views} views to {vis_dir}")

    for idx in range(n_views):
        viewpoint = test_cameras[idx]
        render_kwargs = dict(pbr_kwargs)
        render_kwargs["iteration"] = args.diffuse_iteration + 1  # ensure non-diffuse

        render_pkg = render_fn(
            viewpoint, gaussians, pipe, background,
            is_training=False, dict_params=render_kwargs)

        # Save outputs
        view_dir = os.path.join(vis_dir, f"view_{viewpoint.image_name}")
        os.makedirs(view_dir, exist_ok=True)

        render_img = torch.clamp(render_pkg["render"], 0.0, 1.0)
        gt_img = torch.clamp(viewpoint.original_image.cuda(), 0.0, 1.0)
        depth = render_pkg["depth"]
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        opacity = torch.clamp(render_pkg["opacity"], 0.0, 1.0)
        normal = torch.clamp(render_pkg.get("normal", torch.zeros_like(render_img)) * 0.5 + 0.5, 0.0, 1.0)
        pseudo_normal = torch.clamp(render_pkg.get("pseudo_normal", torch.zeros_like(render_img)) * 0.5 + 0.5, 0.0, 1.0)

        torchvision.utils.save_image(render_img, os.path.join(view_dir, "render.png"))
        torchvision.utils.save_image(gt_img, os.path.join(view_dir, "gt.png"))
        torchvision.utils.save_image(depth_norm, os.path.join(view_dir, "depth.png"))
        torchvision.utils.save_image(opacity, os.path.join(view_dir, "opacity.png"))
        torchvision.utils.save_image(normal, os.path.join(view_dir, "normal.png"))
        torchvision.utils.save_image(pseudo_normal, os.path.join(view_dir, "pseudo_normal.png"))

        if is_pbr and "pbr" in render_pkg:
            pbr_img = torch.clamp(render_pkg["pbr"], 0.0, 1.0)
            torchvision.utils.save_image(pbr_img, os.path.join(view_dir, "pbr.png"))

        vis_dict = render_pkg.get("vis_dict", {})
        for key in ["roughness", "metallic", "base_color"]:
            if key in vis_dict:
                torchvision.utils.save_image(
                    torch.clamp(vis_dict[key], 0.0, 1.0),
                    os.path.join(view_dir, f"{key}.png"))

        if is_equirect:
            cubemap_dir = os.path.join(view_dir, "cubemap")
            os.makedirs(cubemap_dir, exist_ok=True)
            face_names = ["posx", "negx", "posy", "negy", "posz", "negz"]
            cubemap = latlong_to_cubemap_equirect(
                render_img.permute(1, 2, 0), [512, 512])
            for face_idx in range(6):
                face_img = cubemap[face_idx].permute(2, 0, 1)
                torchvision.utils.save_image(
                    face_img, os.path.join(cubemap_dir, f"{face_names[face_idx]}.png"))

        print(f"  [{idx+1}/{n_views}] {viewpoint.image_name} saved")

    print(f"\nDone! Results in {vis_dir}")


if __name__ == "__main__":
    main()
