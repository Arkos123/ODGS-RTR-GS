"""
将 MTPano 输出的法线从 MTPano 规范空间转换到 RTR-GS COLMAP 世界空间。

坐标系链路 (通过多方向实测验证):
  MTPano 空间 (X,Y,Z)
      ↓ X←-Z, Y←-Y, Z←X
  COLMAP view 空间 (+Y下, +Z前)
      ↓ c2w_rot (= rotation)  per-image
  COLMAP world 空间 (最终)

注意: c2w_rot 使用 OpenMVG 原始 rotation 矩阵 (不是 .T),
      RTR-GS 管线中经过 getWorld2View(R.T) + transpose + inverse
      链后等价于使用 rotation 本身。

用法:
    # 转换所有法线，需要 OpenMVG 数据集路径
    python script/convert_mtpano_normal.py \\
        --input_dir data/OmniBlender/barbershop/mtpano_results \\
        --dataset data/OmniBlender/barbershop

    # 仅预览
    python script/convert_mtpano_normal.py --input_dir ... --dataset ... --dry-run
"""
import argparse
import json
import os
import glob
import numpy as np
import cv2


def load_openmvg_cameras(dataset_path: str):
    """从 OpenMVG 格式数据集中加载各视图的 c2w 旋转矩阵

    Returns:
        dict: {filename_stem: c2w_rot_3x3}  (如 {"0": ndarray[3,3], ...})
    """
    ext_path = os.path.join(dataset_path, "data_extrinsics.json")
    views_path = os.path.join(dataset_path, "data_views.json")

    if not os.path.exists(ext_path) or not os.path.exists(views_path):
        raise FileNotFoundError(f"OpenMVG 数据文件不存在: {ext_path} 或 {views_path}")

    with open(ext_path) as f:
        ext_data = json.load(f)
    with open(views_path) as f:
        views_data = json.load(f)

    # 构建 {id_pose: c2w_rot} 映射
    # 注意: RTR-GS 管线中 c2w_rot = rotation (原始 OpenMVG 矩阵)
    # 经过 getWorld2View(R.T) + transpose + inverse 链后等价于 rotation
    pose_to_c2w = {}
    for ex in ext_data["extrinsics"]:
        pid = ex["key"]
        c2w_rot = np.array(ex["value"]["rotation"], dtype=np.float64)
        pose_to_c2w[pid] = c2w_rot

    # 构建 {filename_stem: c2w_rot} 映射
    name_to_c2w = {}
    for v in views_data["views"]:
        data = v["value"]["ptr_wrapper"]["data"]
        stem = os.path.splitext(data["filename"])[0]
        pid = data["id_pose"]
        if pid in pose_to_c2w:
            name_to_c2w[stem] = pose_to_c2w[pid]
        else:
            print(f"  [WARN] pose {pid} 无对应 extrinsic (file: {data['filename']})")

    return name_to_c2w


def convert_normal(normal_mtpano: np.ndarray, c2w_rot: np.ndarray | None = None) -> np.ndarray:
    """MTPano 法线 → COLMAP 世界空间法线

    MTPano 空间到 COLMAP view 空间除了 Y/Z 翻转外，还有 X/Z 交换，
    实测验证通过对比四个方向的墙面和地板/天花板确定。

    Args:
        normal_mtpano: [3, H, W] float32, MTPano 空间, [-1, 1], 向外
        c2w_rot: [3, 3] camera-to-world 旋转矩阵 (rotation, 不是 rotation.T)

    Returns:
        [3, H, W] float32, COLMAP world 空间, 向外法线
    """
    # Step 1: 坐标变换: X←-Z, Y←-Y, Z←X
    # 这是经过实测验证的 MTPano → COLMAP view 空间转换
    n = np.zeros_like(normal_mtpano)
    n[0] = -normal_mtpano[2]  # X ← -Z
    n[1] = -normal_mtpano[1]  # Y ← -Y
    n[2] = normal_mtpano[0]   # Z ← X

    # Step 2: COLMAP view → COLMAP world (用 rotation, 不是 rotation.T)
    if c2w_rot is not None:
        n = (c2w_rot @ n.reshape(3, -1)).reshape(n.shape)

    # Step 3: 归一化
    norm = np.linalg.norm(n, axis=0, keepdims=True)
    n = n / np.clip(norm, 1e-8, None)

    return n


def save_normal_vis(normal: np.ndarray, path: str):
    """将法线保存为可视化 PNG"""
    vis = (normal * 0.5 + 0.5).clip(0, 1)
    vis = (vis.transpose(1, 2, 0) * 255).astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, vis)


def process_dir(input_dir: str, dataset: str | None, output_dir: str | None = None,
                dry_run: bool = False):
    # 加载相机位姿
    name_to_c2w = {}
    if dataset:
        name_to_c2w = load_openmvg_cameras(dataset)
        print(f"已加载 {len(name_to_c2w)} 个相机位姿")

    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    # 找所有 *_normal.npy
    pattern = os.path.join(input_dir, "*", "*_normal.npy")
    pattern2 = os.path.join(input_dir, "*_normal.npy")
    npy_files = sorted(set(glob.glob(pattern) + glob.glob(pattern2)))

    if not npy_files:
        print(f"[WARN] 未找到 *_normal.npy 文件")
        return

    print(f"找到 {len(npy_files)} 个法线文件")

    for src_path in npy_files:
        rel = os.path.relpath(os.path.dirname(src_path), input_dir)
        basename = os.path.basename(src_path).replace("_normal.npy", "")

        out_subdir = output_dir if rel == "." else os.path.join(output_dir, rel)
        os.makedirs(out_subdir, exist_ok=True)

        out_npy = os.path.join(out_subdir, f"{basename}_normal_colmap.npy")
        out_png = os.path.join(out_subdir, f"{basename}_normal_colmap_vis.png")

        normal_mtpano = np.load(src_path)  # [3, H, W]
        h, w = normal_mtpano.shape[1:]

        # 获取该图像的 c2w
        c2w_rot = name_to_c2w.get(basename)
        if dataset and c2w_rot is None:
            print(f"  ⚠ {basename}: 未找到相机位姿，跳过")
            continue

        if dry_run:
            print(f"  [DRY-RUN] {basename}: {normal_mtpano.shape}, "
                  f"c2w={'有' if c2w_rot is not None else '无'}")
            continue

        # 转换
        normal_colmap = convert_normal(normal_mtpano, c2w_rot)

        # 保存
        np.save(out_npy, normal_colmap)
        save_normal_vis(normal_colmap, out_png)
        print(f"  ✓ {basename}: ({h}x{w}) → {os.path.relpath(out_npy)}")

    if not dry_run:
        print(f"\n完成！结果在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="将 MTPano 法线转换到 COLMAP 世界空间")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="MTPano 结果目录（含 *_normal.npy）")
    parser.add_argument("--dataset", type=str, default=None,
                        help="OpenMVG 数据集路径（含 data_extrinsics.json 和 data_views.json）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录（默认同 input_dir）")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅预览，不保存文件")
    args = parser.parse_args()

    assert os.path.isdir(args.input_dir), f"input_dir 不存在: {args.input_dir}"
    if args.dataset:
        assert os.path.isdir(args.dataset), f"dataset 不存在: {args.dataset}"

    process_dir(args.input_dir, args.dataset, args.output_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
