"""
ODGS .ply → RTR-GS .ply 转换脚本

将 ODGS 训练输出的 .ply 文件转换为 RTR-GS 格式，
将 RTR-GS 扩展属性初始化为默认值。

用法:
    python script/odgs2rtrgs.py <odgs_ply_path> <output_path>
    python script/odgs2rtrgs.py <odgs_ply_path> <output_path> --sh_degree 3

也可以直接传入模型路径和迭代次数:
    python script/odgs2rtrgs.py -m <odgs_model_path> -i 30000 -o <output_path>
"""
import argparse
import os
import sys
import numpy as np
from plyfile import PlyData, PlyElement


def inverse_sigmoid(x):
    return np.log(x / (1 - x))


def load_odgs_ply(path, sh_degree=3):
    """加载 ODGS 格式的 .ply 文件"""
    plydata = PlyData.read(path)

    xyz = np.stack([
        np.asarray(plydata.elements[0]["x"]),
        np.asarray(plydata.elements[0]["y"]),
        np.asarray(plydata.elements[0]["z"]),
    ], axis=1)

    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = sorted(
        [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")],
        key=lambda x: int(x.split('_')[-1]),
    )
    assert len(extra_f_names) == 3 * (sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (sh_degree + 1) ** 2 - 1))

    scale_names = sorted(
        [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")],
        key=lambda x: int(x.split('_')[-1]),
    )
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = sorted(
        [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")],
        key=lambda x: int(x.split('_')[-1]),
    )
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    return xyz, features_dc, features_extra, opacities, scales, rots


def convert_odgs_to_rtrgs(xyz, features_dc, features_extra, opacities, scales, rots, sh_degree=3):
    """将 ODGS 属性转换为 RTR-GS 格式，添加默认的扩展属性"""
    N = xyz.shape[0]
    n_sh_coeffs = (sh_degree + 1) ** 2

    # ODGS 的 SH → RTR-GS 的 shs
    # ODGS: features_dc=[N,3,1], features_extra=[N,3,(d+1)²-1]
    # RTR-GS: _shs_dc=[N,1,3], _shs_rest=[N,(d+1)²-1,3]
    # 所以需要对调维度
    rtr_shs_dc = features_dc.transpose(0, 2, 1).copy()      # [N, 1, 3]
    rtr_shs_rest = features_extra.transpose(0, 2, 1).copy()  # [N, (d+1)²-1, 3]

    # RTR-GS 扩展属性初始化为默认值
    diffuse_tint = np.zeros((N, 3), dtype=np.float32)
    specular_tint = np.zeros((N, 3), dtype=np.float32)
    ref_tint = np.zeros((N, 3), dtype=np.float32)
    ref_strength = np.full((N, 1), inverse_sigmoid(0.01), dtype=np.float32)
    ref_roughness = np.full((N, 1), inverse_sigmoid(0.65), dtype=np.float32)
    specular_feature = np.zeros((N, 10), dtype=np.float32)
    diffuse_transfer_dc = np.zeros((N, 1, 1), dtype=np.float32)
    diffuse_transfer_rest = np.zeros((N, n_sh_coeffs - 1, 1), dtype=np.float32)

    # 拼接所有属性：顺序需与 construct_list_of_attributes 保持一致
    attributes_list = [
        xyz,                                              # 3
        rtr_shs_dc.reshape(N, -1),                        # 3
        rtr_shs_rest.reshape(N, -1),                      # 3 * ((d+1)² - 1)
        diffuse_tint,                                     # 3
        specular_tint,                                    # 3
        ref_tint,                                         # 3
        ref_strength,                                     # 1
        ref_roughness,                                    # 1
        specular_feature,                                 # 10
        diffuse_transfer_dc.reshape(N, -1),               # 1
        diffuse_transfer_rest.reshape(N, -1),             # (d+1)² - 1
        opacities.reshape(N, -1),                         # 1
        scales,                                           # 3
        rots,                                             # 4
    ]

    return np.concatenate(attributes_list, axis=1)


def construct_attribute_names(sh_degree=3):
    """构造 RTR-GS PLY 的属性名列表，与 construct_list_of_attributes 一致"""
    n_sh_coeffs = (sh_degree + 1) ** 2

    names = ['x', 'y', 'z']
    for i in range(3):
        names.append(f'f_dc_{i}')
    for i in range(3 * (n_sh_coeffs - 1)):
        names.append(f'f_rest_{i}')
    for i in range(3):
        names.append(f'diffuse_tint_{i}')
    for i in range(3):
        names.append(f'specular_tint_{i}')
    for i in range(3):
        names.append(f'ref_tint_{i}')
    names.append('ref_strength')
    names.append('ref_roughness')
    for i in range(10):
        names.append(f'specular_feature_{i}')
    for i in range(1):
        names.append(f'diffuse_transfer_dc_{i}')
    for i in range(n_sh_coeffs - 1):
        names.append(f'diffuse_transfer_rest_{i}')
    names.append('opacity')
    for i in range(3):
        names.append(f'scale_{i}')
    for i in range(4):
        names.append(f'rot_{i}')

    return names


def save_rtrgs_ply(path, attributes, sh_degree=3):
    """保存为 RTR-GS 格式的 .ply 文件"""
    names = construct_attribute_names(sh_degree)
    dtype_full = [(name, 'f4') for name in names]

    elements = np.empty(attributes.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
    print(f"Saved RTR-GS PLY to {path} ({attributes.shape[0]} points)")


def main():
    parser = argparse.ArgumentParser(description="Convert ODGS .ply to RTR-GS .ply")
    parser.add_argument("input_ply", nargs="?", type=str, default=None,
                        help="Path to ODGS .ply file")
    parser.add_argument("output_ply", nargs="?", type=str, default=None,
                        help="Path to output RTR-GS .ply file")
    parser.add_argument("--sh_degree", type=int, default=3,
                        help="SH degree used in ODGS training (default: 3)")
    parser.add_argument("-m", "--model_path", type=str, default=None,
                        help="ODGS model path (alternative to input_ply)")
    parser.add_argument("-i", "--iteration", type=int, default=None,
                        help="Iteration number (used with -m)")
    parser.add_argument("-o", "--output_dir", type=str, default=None,
                        help="Output directory (used with -m)")

    args = parser.parse_args()

    if args.model_path is not None:
        if args.iteration is None:
            print("Error: --iteration (-i) is required when using --model_path (-m)")
            sys.exit(1)
        input_ply = os.path.join(args.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
        if not os.path.exists(input_ply):
            input_ply = os.path.join(args.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
        if not os.path.exists(input_ply):
            print(f"Error: Could not find PLY at {input_ply}")
            sys.exit(1)
        if args.output_dir:
            output_ply = os.path.join(args.output_dir, "point_cloud.ply")
        else:
            output_ply = os.path.join(args.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud_rtrgs.ply")
    else:
        if args.input_ply is None:
            print("Error: either input_ply or --model_path is required")
            sys.exit(1)
        input_ply = args.input_ply
        output_ply = args.output_ply or input_ply.replace(".ply", "_rtrgs.ply")

    if not os.path.exists(input_ply):
        print(f"Error: input PLY not found at {input_ply}")
        sys.exit(1)

    print(f"Loading ODGS PLY from: {input_ply}")
    xyz, f_dc, f_rest, opacities, scales, rots = load_odgs_ply(input_ply, args.sh_degree)
    print(f"Loaded {xyz.shape[0]} points")

    print("Converting to RTR-GS format...")
    attributes = convert_odgs_to_rtrgs(xyz, f_dc, f_rest, opacities, scales, rots, args.sh_degree)

    os.makedirs(os.path.dirname(os.path.abspath(output_ply)), exist_ok=True)
    save_rtrgs_ply(output_ply, attributes, args.sh_degree)


if __name__ == "__main__":
    main()
