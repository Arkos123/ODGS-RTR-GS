"""
检查 RTR-GS 格式 .ply 点云文件的高斯点数量及属性信息

用法:
    python script/check_rtrgs_ply.py <ply_path>
    python script/check_rtrgs_ply.py lab_output/360Roam/base/odgs/iteration_30000/point_cloud_rtrgs.ply
"""
import argparse
import os
import numpy as np
from plyfile import PlyData


def check_rtrgs_ply(path):
    """检查 RTR-GS PLY 文件的高斯点数量和属性"""
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        return

    print(f"\n{'='*60}")
    print(f"Checking RTR-GS PLY: {path}")
    print(f"{'='*60}")

    plydata = PlyData.read(path)
    vertex_element = plydata.elements[0]

    num_points = vertex_element.count
    properties = [p.name for p in vertex_element.properties]
    num_properties = len(properties)

    print(f"\n[Basic Info]")
    print(f"  Number of Gaussians (points): {num_points:,}")
    print(f"  Number of properties per point: {num_properties}")
    print(f"  Total values in file: {num_points * num_properties:,}")

    print(f"\n[Property Names] ({num_properties} total)")
    for i, name in enumerate(properties):
        print(f"  {i+1:3d}. {name}")

    print(f"\n[Property Categories]")
    categories = {
        'Position': [p for p in properties if p in ['x', 'y', 'z']],
        'SH DC (f_dc)': [p for p in properties if p.startswith('f_dc_')],
        'SH Rest (f_rest)': [p for p in properties if p.startswith('f_rest_')],
        'Diffuse Tint': [p for p in properties if p.startswith('diffuse_tint_')],
        'Specular Tint': [p for p in properties if p.startswith('specular_tint_')],
        'Reflection Tint': [p for p in properties if p.startswith('ref_tint_')],
        'Reflection Strength': [p for p in properties if p == 'ref_strength'],
        'Reflection Roughness': [p for p in properties if p == 'ref_roughness'],
        'Specular Feature': [p for p in properties if p.startswith('specular_feature_')],
        'Diffuse Transfer DC': [p for p in properties if p.startswith('diffuse_transfer_dc_')],
        'Diffuse Transfer Rest': [p for p in properties if p.startswith('diffuse_transfer_rest_')],
        'Opacity': [p for p in properties if p == 'opacity'],
        'Scale': [p for p in properties if p.startswith('scale_')],
        'Rotation': [p for p in properties if p.startswith('rot_')],
    }

    for cat_name, cat_props in categories.items():
        if cat_props:
            print(f"  {cat_name:25s}: {len(cat_props)} properties")

    sh_degree = None
    f_rest_count = len([p for p in properties if p.startswith('f_rest_')])
    if f_rest_count > 0:
        # f_rest_count = 3 * ((sh_degree + 1)^2 - 1)
        # => (sh_degree + 1)^2 = f_rest_count / 3 + 1
        n_sh_coeffs = f_rest_count // 3 + 1
        sh_degree = int(np.sqrt(n_sh_coeffs)) - 1
        print(f"\n[Inferred SH Degree]")
        print(f"  f_rest properties: {f_rest_count}")
        print(f"  SH coefficients per channel: {n_sh_coeffs}")
        print(f"  SH degree: {sh_degree}")

    print(f"\n[Memory Estimate]")
    bytes_per_float = 4
    total_bytes = num_points * num_properties * bytes_per_float
    print(f"  Raw data size: {total_bytes / (1024**2):.2f} MB")
    print(f"  Raw data size: {total_bytes / (1024**3):.2f} GB")

    print(f"\n{'='*60}")
    print(f"Summary: {num_points:,} Gaussians in {os.path.basename(path)}")
    print(f"{'='*60}\n")

    return num_points


def main():
    parser = argparse.ArgumentParser(description="Check RTR-GS PLY point cloud")
    parser.add_argument("ply_path", type=str, help="Path to RTR-GS .ply file")
    args = parser.parse_args()

    check_rtrgs_ply(args.ply_path)


if __name__ == "__main__":
    main()
