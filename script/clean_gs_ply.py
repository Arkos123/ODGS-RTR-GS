"""
Post-processing: 从训练好的 checkpoint 中清除 floater 高斯点。

Floater 特征:
  - 不透明度极低 (opacity < threshold)
  - 各向异性极高 (max_scale / min_scale > threshold)
  - 世界空间尺度过大 (max_scale > ratio * scene_extent)

用法:
    # 从 checkpoint 清洗并保存为 PLY
    python script/clean_gs_ply.py lab_output/scene/checkpoint/chkpnt21000.pth \\
        --output scene_cleaned.ply

    # 仅按透明度清洗
    python script/clean_gs_ply.py chkpnt.pth --opacity 0.02 --anisotropy 0 --world_size_ratio 0

    # 查看各指标统计（不实际清洗）
    python script/clean_gs_ply.py chkpnt.pth --dry-run
"""
import argparse
import os
import sys
import torch
import numpy as np
from plyfile import PlyData, PlyElement


def inverse_sigmoid(x):
    return np.log(x / (1 - x))


def extract_gaussians(model_args, use_pbr):
    """从 captured model_args 中提取所有高斯属性。"""
    idx = 0
    active_sh_degree = model_args[idx]; idx += 1
    _xyz = model_args[idx]; idx += 1
    _shs_dc = model_args[idx]; idx += 1
    _shs_rest = model_args[idx]; idx += 1
    _diffuse_tint = model_args[idx]; idx += 1
    _specular_tint = model_args[idx]; idx += 1
    _ref_tint = model_args[idx]; idx += 1
    _ref_strength = model_args[idx]; idx += 1
    _ref_roughness = model_args[idx]; idx += 1
    _specular_feature = model_args[idx]; idx += 1
    _diffuse_transfer_dc = model_args[idx]; idx += 1
    _diffuse_transfer_rest = model_args[idx]; idx += 1
    _scaling = model_args[idx]; idx += 1
    _rotation = model_args[idx]; idx += 1
    _opacity = model_args[idx]; idx += 1
    max_radii2D = model_args[idx]; idx += 1
    weights_accum = model_args[idx]; idx += 1
    xyz_gradient_accum = model_args[idx]; idx += 1
    denom = model_args[idx]; idx += 1
    opt_dict = model_args[idx]; idx += 1
    spatial_lr_scale = model_args[idx]; idx += 1

    extra = {}
    if use_pbr and len(model_args) > idx:
        extra['base_color'] = model_args[idx]; idx += 1
        extra['roughness'] = model_args[idx]; idx += 1
        extra['metallic'] = model_args[idx]; idx += 1
        extra['incidents_dc'] = model_args[idx]; idx += 1
        extra['incidents_rest'] = model_args[idx]; idx += 1

    return {
        'active_sh_degree': active_sh_degree,
        'xyz': _xyz, 'shs_dc': _shs_dc, 'shs_rest': _shs_rest,
        'diffuse_tint': _diffuse_tint, 'specular_tint': _specular_tint,
        'ref_tint': _ref_tint, 'ref_strength': _ref_strength,
        'ref_roughness': _ref_roughness, 'specular_feature': _specular_feature,
        'diffuse_transfer_dc': _diffuse_transfer_dc,
        'diffuse_transfer_rest': _diffuse_transfer_rest,
        'scaling': _scaling, 'rotation': _rotation, 'opacity': _opacity,
        'max_radii2D': max_radii2D, 'weights_accum': weights_accum,
        'xyz_gradient_accum': xyz_gradient_accum, 'denom': denom,
        'opt_dict': opt_dict, 'spatial_lr_scale': spatial_lr_scale,
        **extra,
    }


def compute_clean_mask(gaussians, args, device='cuda'):
    """计算需要修剪的高斯 mask。返回 (mask, stats_dict)。"""
    xyz = gaussians['xyz']
    n = xyz.shape[0]
    if n == 0:
        return torch.zeros(0, dtype=torch.bool, device=device), {}

    # 移至 GPU
    xyz = xyz.to(device)
    scales = gaussians['scaling'].to(device)
    opacity = torch.sigmoid(gaussians['opacity'].to(device)).squeeze()
    rotation = gaussians['rotation'].to(device)

    # 场景范围估计
    center = xyz.mean(dim=0)
    dist_from_center = torch.norm(xyz - center, dim=1)
    scene_extent = float(dist_from_center.quantile(0.95).item() * 2)
    scene_extent = max(scene_extent, 1.0)

    max_scale = scales.max(dim=1).values
    min_scale = scales.min(dim=1).values.clamp_min(1e-8)
    anisotropy = max_scale / min_scale

    mask = torch.zeros(n, dtype=torch.bool, device=device)
    stats = {}

    if args.opacity > 0:
        low_op = opacity < args.opacity
        stats['low_opacity'] = low_op.sum().item()
        if not args.dry_run:
            mask = mask | low_op

    if args.anisotropy > 0:
        high_aniso = anisotropy > args.anisotropy
        stats['high_anisotropy'] = high_aniso.sum().item()
        if not args.dry_run:
            mask = mask | high_aniso

    if args.world_size_ratio > 0:
        large = max_scale > args.world_size_ratio * scene_extent
        stats['large_world'] = large.sum().item()
        if not args.dry_run:
            mask = mask | large

    if args.outlier_std > 0:
        # Z-score 异常点检测
        mean_xyz = xyz.mean(dim=0)
        std_xyz = xyz.std(dim=0).clamp_min(1e-8)
        z_scores = ((xyz - mean_xyz).abs() / std_xyz).max(dim=1).values
        outlier = z_scores > args.outlier_std
        stats['outlier'] = outlier.sum().item()
        if not args.dry_run:
            mask = mask | outlier

    stats['total'] = n
    stats['to_prune'] = mask.sum().item()
    stats['prune_ratio'] = mask.sum().item() / max(n, 1) * 100
    stats['remaining'] = n - mask.sum().item()
    stats['scene_extent'] = scene_extent

    if not args.dry_run:
        return mask, stats
    return None, stats


def construct_ply_attributes(gaussians, keep_mask, sh_degree=3):
    """构建 PLY 属性数组（与 GaussianModel.construct_list_of_attributes 兼容）。"""
    device = gaussians['xyz'].device

    def maybe_mask(t):
        t = t.detach().cpu().numpy()
        return t[keep_mask.cpu().numpy()]

    xyz = maybe_mask(gaussians['xyz'])
    opacities = inverse_sigmoid(
        np.clip(torch.sigmoid(gaussians['opacity'])[keep_mask].cpu().numpy(), 1e-8, 1 - 1e-8)
    ).reshape(-1, 1)

    # SH
    f_dc = gaussians['shs_dc'][keep_mask].detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy()
    f_rest = gaussians['shs_rest'][keep_mask].detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy()

    scales = maybe_mask(gaussians['scaling'])
    rots = maybe_mask(gaussians['rotation'])

    # RTR-GS 扩展属性
    diffuse_tint = maybe_mask(gaussians['diffuse_tint'])
    specular_tint = maybe_mask(gaussians['specular_tint'])
    ref_tint = maybe_mask(gaussians['ref_tint'])
    ref_strength = maybe_mask(gaussians['ref_strength'])
    ref_roughness = maybe_mask(gaussians['ref_roughness'])
    specular_feature = maybe_mask(gaussians['specular_feature'])
    diffuse_transfer_dc = gaussians['diffuse_transfer_dc'][keep_mask].detach().flatten(start_dim=1).cpu().numpy()
    diffuse_transfer_rest = gaussians['diffuse_transfer_rest'][keep_mask].detach().flatten(start_dim=1).cpu().numpy()

    # Concatenate in the same order as construct_list_of_attributes
    attr_list = [
        xyz,
        f_dc,
        f_rest,
        diffuse_tint,
        specular_tint,
        ref_tint,
        ref_strength,
        ref_roughness,
        specular_feature,
        diffuse_transfer_dc,
        diffuse_transfer_rest,
        opacities,
        scales,
        rots,
    ]

    if 'base_color' in gaussians:
        attr_list.extend([
            maybe_mask(gaussians['base_color']),
            maybe_mask(gaussians['roughness']),
            maybe_mask(gaussians['metallic']),
            gaussians['incidents_dc'][keep_mask].detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy(),
            gaussians['incidents_rest'][keep_mask].detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy(),
        ])

    return np.concatenate(attr_list, axis=1)


def build_attribute_names(sh_degree=3, use_pbr=False):
    """构建 PLY 属性名列表。"""
    n_sh_coeffs = (sh_degree + 1) ** 2
    names = ['x', 'y', 'z']
    for i in range(3):
        names.append(f'f_dc_{i}')
    for i in range(3 * (n_sh_coeffs - 1)):
        names.append(f'f_rest_{i}')
    for i in range(3):
        names.extend([f'diffuse_tint_{i}', f'specular_tint_{i}', f'ref_tint_{i}'])
    names.append('ref_strength')
    names.append('ref_roughness')
    for i in range(10):
        names.append(f'specular_feature_{i}')
    for i in range(1):
        names.append('diffuse_transfer_dc_0')
    for i in range(n_sh_coeffs - 1):
        names.append(f'diffuse_transfer_rest_{i}')
    names.append('opacity')
    for i in range(3):
        names.append(f'scale_{i}')
    for i in range(4):
        names.append(f'rot_{i}')
    if use_pbr:
        for i in range(3):
            names.extend([f'base_color_{i}', f'roughness_{i}', f'metallic_{i}'])
        for i in range(3):
            names.append(f'incidents_dc_{i}')
        for i in range(3 * (n_sh_coeffs - 1)):
            names.append(f'incidents_rest_{i}')
    return names


def apply_mask_to_model_args(model_args, keep_mask):
    """对 model_args 中的所有高斯属性张量应用 keep_mask，同时清理优化器状态。"""
    masked = list(model_args)  # 转为可变列表
    device = keep_mask.device

    # 需要 mask 的张量索引（所有形状为 [N, ...] 的字段）
    # 索引: 0=active_sh_degree(标量), 1-14=NN参数, 15-18=统计, 19=optimizer, 20=spatial_lr_scale
    #       21-25=PBR属性（仅当 use_pbr=True）
    n_total = keep_mask.shape[0]
    param_indices = [i for i in range(len(masked))
                     if isinstance(masked[i], torch.Tensor)
                     and masked[i].dim() > 0
                     and masked[i].shape[0] == n_total]
    for i in param_indices:
        if isinstance(masked[i], torch.Tensor) and masked[i].shape[0] == keep_mask.shape[0]:
            masked[i] = masked[i][keep_mask.to(device=masked[i].device)].contiguous()

    # 处理优化器状态: 动量同步修剪，保持与参数量一致
    old_opt = masked[19]
    if isinstance(old_opt, dict):
        new_state = {}
        for pid, pstate in old_opt.get('state', {}).items():
            new_pstate = {}
            for key, val in pstate.items():
                # 标量（如 step）直接保留，per-parameter 张量按 keep_mask 索引
                if isinstance(val, torch.Tensor) and val.dim() > 0 and val.shape[0] == keep_mask.shape[0]:
                    val = val[keep_mask.to(device=val.device)].contiguous()
                new_pstate[key] = val
            new_state[pid] = new_pstate
        masked[19] = {
            'state': new_state,
            'param_groups': old_opt.get('param_groups', []),
        }

    return tuple(masked)


def save_checkpoint(output_path, model_args, iteration):
    """保存为 .pth checkpoint 文件。"""
    torch.save((model_args, iteration), output_path)
    # 验证可加载
    verify = torch.load(output_path, map_location='cpu', weights_only=True)
    n = verify[0][1].shape[0] if isinstance(verify, (list, tuple)) else 0
    print(f"Saved: {output_path} ({n} gaussians, iter {iteration})")


def save_ply(path, attributes, names):
    """保存为 PLY 文件。"""
    dtype_full = [(name, 'f4') for name in names]
    elements = np.empty(attributes.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
    print(f"Saved: {path} ({attributes.shape[0]} points)")


def main():
    parser = argparse.ArgumentParser(description="清除训练好的 checkpoint 中的 floater")
    parser.add_argument("checkpoint", type=str, help="Path to .pth checkpoint")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output path (.ply or .pth). Extension determines format.")
    parser.add_argument("--save-as", choices=['ply', 'checkpoint', 'auto'], default='auto',
                        help="Save format: 'ply' (default), 'checkpoint' (.pth with optimizer),"
                             " or 'auto' (infer from --output extension)")
    parser.add_argument("--opacity", type=float, default=0.01,
                        help="Opacity threshold (default: 0.01, ≤0 to disable)")
    parser.add_argument("--anisotropy", type=float, default=15,
                        help="Anisotropy ratio threshold (default: 15, ≤0 to disable)")
    parser.add_argument("--world_size_ratio", type=float, default=0.2,
                        help="Max world size ratio relative to scene extent (default: 0.2, ≤0 to disable)")
    parser.add_argument("--outlier_std", type=float, default=0,
                        help="Z-score outlier threshold (default: 0=disabled)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only show statistics, don't clean")
    parser.add_argument("--sh_degree", type=int, default=3,
                        help="SH degree (default: 3)")

    args = parser.parse_args()

    # 检测是否有 PBR 属性
    ckpt_data = torch.load(args.checkpoint, weights_only=True)
    model_args, iteration = ckpt_data if isinstance(ckpt_data, (list, tuple)) else (ckpt_data, 0)
    use_pbr = len(model_args) > 21
    sh_degree = args.sh_degree

    print(f"Checkpoint: iteration={iteration}, {len(model_args)} fields, PBR={use_pbr}")

    gaussians = extract_gaussians(model_args, use_pbr)
    n = gaussians['xyz'].shape[0]
    print(f"Loaded {n} Gaussians")

    # 计算指标
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mask, stats = compute_clean_mask(gaussians, args, device=device)

    print(f"\nScene extent estimate: {stats['scene_extent']:.3f}")
    if 'low_opacity' in stats:
        print(f"  Low opacity (<{args.opacity}):              {stats['low_opacity']:>8d}")
    if 'high_anisotropy' in stats:
        print(f"  High anisotropy (>{args.anisotropy}):        {stats['high_anisotropy']:>8d}")
    if 'large_world' in stats:
        print(f"  Large world (>{args.world_size_ratio}*ext):   {stats['large_world']:>8d}")
    if 'outlier' in stats:
        print(f"  Outlier (z>{args.outlier_std}):               {stats['outlier']:>8d}")
    print(f"  ─────────────────────────────────────────")
    print(f"  To prune:      {stats['to_prune']:>8d} / {stats['total']} ({stats['prune_ratio']:.1f}%)")
    print(f"  Remaining:     {stats['remaining']:>8d}")

    if args.dry_run:
        print("\nDry run complete. Use --output to save cleaned checkpoint/PLY.")
        return

    if mask.sum() == 0:
        print("Nothing to clean.")
        return

    keep_mask = ~mask

    # 确定输出格式和路径
    save_as = args.save_as
    output_path = args.output
    if output_path is None:
        ckpt_path = os.path.abspath(args.checkpoint)
        ckpt_dir = os.path.dirname(ckpt_path)
        base, ext = os.path.splitext(os.path.basename(ckpt_path))
        stem = base.rsplit('_cleaned', 1)[0]
        out_name = f"{stem}_cleaned{ext}"
        if save_as == 'checkpoint':
            output_path = os.path.join(ckpt_dir, out_name)
        else:
            output_path = os.path.join(ckpt_dir, out_name)
            save_as = 'ply'
    else:
        if save_as == 'auto':
            save_as = 'checkpoint' if output_path.endswith('.pth') else 'ply'

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if save_as == 'checkpoint':
        # 保存为 checkpoint（保留模型结构，优化器动量同步修剪）
        print("Saving checkpoint format (optimizer momentum preserved)...")
        new_model_args = apply_mask_to_model_args(model_args, keep_mask.to('cpu'))
        save_checkpoint(output_path, new_model_args, iteration)
    else:
        # 保存为 PLY
        attributes = construct_ply_attributes(gaussians, keep_mask, sh_degree)
        names = build_attribute_names(sh_degree, use_pbr)
        save_ply(output_path, attributes, names)

    print("Done.")


if __name__ == "__main__":
    main()
