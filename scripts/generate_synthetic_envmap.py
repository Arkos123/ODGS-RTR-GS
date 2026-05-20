import numpy as np
import cv2
import os

def generate_directional_light_hdr(output_path, direction_theta=0.0, direction_phi=np.pi/4,
                                    angular_width=5.0, intensity=10.0, ambient=0.02,
                                    width=1024, height=512):
    """
    生成模拟方向光源的 HDR 环境贴图（等距柱状投影 / LatLong）

    Args:
        output_path:     输出 .hdr 文件路径
        direction_theta: 光源方向——水平角（弧度），0=正前方(-z)，π/2=右侧(+x)，π=后方(+z)
        direction_phi:   光源方向——仰角（弧度），0=正上方(+y)，π/2=水平，π=正下方(-y)
        angular_width:   光源半角宽度（度），越大光斑越散
        intensity:       光源峰值强度（HDR 亮度倍率，>1 表示高亮）
        ambient:         环境光最低亮度
        width:           贴图宽度（经度方向）
        height:          贴图高度（纬度方向）
    """
    # 经纬度网格
    theta = np.linspace(0, 2 * np.pi, width, dtype=np.float32)     # 经度 0→2π
    phi   = np.linspace(np.pi, 0, height, dtype=np.float32)       # 纬度 π→0（从上到下）

    THETA, PHI = np.meshgrid(theta, phi, indexing='xy')

    # 像素 → 单位方向向量（Y-up 坐标系统）
    # x = sin(phi) * cos(theta)
    # y = cos(phi)
    # z = sin(phi) * sin(theta)
    px = np.sin(PHI) * np.cos(THETA)
    py = np.cos(PHI)
    pz = np.sin(PHI) * np.sin(THETA)

    # 光源方向向量
    lx = np.sin(direction_phi) * np.cos(direction_theta)
    ly = np.cos(direction_phi)
    lz = np.sin(direction_phi) * np.sin(direction_theta)

    # 点积 = cos(angular_distance)
    dot = np.clip(px * lx + py * ly + pz * lz, -1.0, 1.0)
    angular_dist = np.arccos(dot)  # 弧度

    sigma = np.radians(angular_width)
    gauss = np.exp(-0.5 * (angular_dist / sigma) ** 2)

    hdr = np.stack([gauss, gauss, gauss], axis=-1).astype(np.float32)
    hdr = hdr * intensity + ambient

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(hdr, cv2.COLOR_RGB2BGR))
    print(f"Generated: {output_path}  (peak={intensity:.1f}, width={angular_width}°, ambient={ambient})")


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "env_maps")

    # ---- 预设：几个不同方向的方向光 ----

    presets = [
        # (文件名, theta, phi, angular_width, intensity, ambient, 说明)
        ("directional_front.hdr",      0.0,           np.pi/2, 5,  20.0, 0.02, "正前方 (theta=0, 水平)"),
        ("directional_top.hdr",        0.0,           0.2,     5,  20.0, 0.02, "正上方"),
        ("directional_left.hdr",       -np.pi/2,      np.pi/2, 5,  20.0, 0.02, "左侧"),
        ("directional_right.hdr",      np.pi/2,       np.pi/2, 5,  20.0, 0.02, "右侧"),
        ("directional_behind.hdr",     np.pi,         np.pi/2, 5,  20.0, 0.02, "正后方"),
        ("directional_front_top.hdr",  0.0,           np.pi/6, 5,  40.0, 0.05, "前上方 60°"),
        ("directional_front_wide.hdr", 0.0,           np.pi/2, 15, 15.0, 0.05, "正前方宽光源 (15°)"),
        ("directional_front_soft.hdr", 0.0,           np.pi/2, 30, 8.0,  0.10, "正前方柔和光源 (30°)"),
    ]

    for fname, theta, phi, aw, intens, amb, desc in presets:
        out = os.path.join(base_dir, fname)
        generate_directional_light_hdr(
            output_path=out,
            direction_theta=theta,
            direction_phi=phi,
            angular_width=aw,
            intensity=intens,
            ambient=amb,
        )
        print(f"  [{desc}]")
    print("\nDone! All files saved to:", base_dir)
