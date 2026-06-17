import numpy as np
import cv2
import json
import os
import shutil
from pathlib import Path
from argparse import ArgumentParser


def build_equi_to_face_map(equi_w, equi_h, face_size, face_rot, border_replicate=False):
    """Build OpenCV remap arrays for equirectangular → cube face conversion.

    Args:
        equi_w, equi_h: equirectangular image dimensions
        face_size: output cube face size (square)
        face_rot: 3x3 rotation matrix that maps camera +z to face forward direction
        border_replicate: if True, use BORDER_REPLICATE near edges; else BORDER_WRAP

    Returns:
        map_x, map_y: float32 arrays for cv2.remap
    """
    xs = np.arange(face_size, dtype=np.float32)
    ys = np.arange(face_size, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)

    u = (xx + 0.5) / face_size * 2.0 - 1.0
    v = (yy + 0.5) / face_size * 2.0 - 1.0

    dx = u
    dy = -v
    dz = np.ones_like(u)

    norm = np.sqrt(dx * dx + dy * dy + dz * dz)
    dx /= norm
    dy /= norm
    dz /= norm

    dirs = np.stack([dx, dy, dz], axis=-1)
    dirs_cam = dirs @ face_rot.T

    lon = np.arctan2(dirs_cam[..., 0], dirs_cam[..., 2])
    lat = np.arcsin(np.clip(dirs_cam[..., 1], -1.0, 1.0))

    map_x = (lon / (2.0 * np.pi) + 0.5) * equi_w
    map_y = (0.5 - lat / np.pi) * equi_h

    return map_x.astype(np.float32), map_y.astype(np.float32)


def extract_cube_face(equi_img, face_size, face_rot):
    """Extract one cube face from an equirectangular image.

    Args:
        equi_img: (H, W, 3) uint8 or float RGB image
        face_size: output square face size
        face_rot: 3x3 rotation matrix

    Returns:
        (face_size, face_size, 3) uint8 image
    """
    h, w = equi_img.shape[:2]
    map_x, map_y = build_equi_to_face_map(w, h, face_size, face_rot)
    flag = cv2.INTER_LANCZOS4 if equi_img.dtype == np.uint8 else cv2.INTER_LINEAR
    face = cv2.remap(equi_img, map_x, map_y, flag, borderMode=cv2.BORDER_WRAP)
    return face


def save_face_as_png(face, output_path):
    """Save cube face as RGBA PNG (fully opaque alpha).

    Note: face is in RGB order, but cv2.imwrite expects BGR/BGRA.
    We convert RGB→BGRA first so the saved PNG has correct color channels.
    """
    bgra = cv2.cvtColor(face, cv2.COLOR_RGB2BGRA)
    cv2.imwrite(output_path, bgra, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def main():
    parser = ArgumentParser(description="Convert OpenMVG equirectangular dataset to Blender/NeRF-Synthetic format with cube faces.")
    parser.add_argument("--source_path", "-s", required=True,
                        help="Path to OpenMVG dataset (containing data_extrinsics.json, data_views.json, images/)")
    parser.add_argument("--output_path", "-o", required=True,
                        help="Output path for Blender-format dataset")
    parser.add_argument("--face_size", type=int, default=1024,
                        help="Cube face size in pixels (default: 1024)")
    parser.add_argument("--faces", nargs="+", default=["F", "B", "L", "R"],
                        help="Cube faces to extract (default: F B L R)")
    parser.add_argument("--step", type=int, default=1,
                        help="Only process every N-th camera view (default: 1 = all)")
    parser.add_argument("--pitch", nargs="*", type=float, default=[0],
                        help="Pitch offset(s) in degrees. Multiple values cycle per view.")
    parser.add_argument("--yaw", nargs="*", type=float, default=[0],
                        help="Yaw offset(s) in degrees. Multiple values cycle per view.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output directory")
    args = parser.parse_args()

    src = Path(args.source_path)
    dst = Path(args.output_path)

    if dst.exists():
        if args.force:
            shutil.rmtree(dst)
        else:
            raise FileExistsError(f"Output path {dst} already exists. Use --force to overwrite.")

    # ── Face rotation matrices ──────────────────────────────────────────
    # Each maps camera forward (+z) to the face's forward direction
    # Camera convention: +x right, +y up, +z forward
    face_rots = {
        "F": np.eye(3, dtype=np.float64),
        "B": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64),
        "L": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),
        "R": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64),
        "U": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64),
        "D": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64),
    }

    selected_faces = args.faces
    for f in selected_faces:
        assert f in face_rots, f"Unknown face {f}, choose from {list(face_rots.keys())}"
    print(f"Using {len(selected_faces)} faces: {selected_faces}")

    # ── 1. Parse OpenMVG data ───────────────────────────────────────────
    print("Reading OpenMVG data...")

    views_path = src / "data_views.json"
    with open(views_path) as f:
        views_data = json.load(f)
    cam_file_dict = {}
    for view in views_data["views"]:
        cam_file_dict[view["key"]] = view["value"]["ptr_wrapper"]["data"]["filename"]

    extrinsics_path = src / "data_extrinsics.json"
    with open(extrinsics_path) as f:
        ext_data = json.load(f)

    train_names = set()
    test_names = set()
    train_file = src / "train.txt"
    test_file = src / "test.txt"
    if train_file.exists():
        with open(train_file) as f:
            train_names = set(f.read().splitlines())
    if test_file.exists():
        with open(test_file) as f:
            test_names = set(f.read().splitlines())

    print(f"Found {len(ext_data['extrinsics'])} camera views")
    print(f"  Train: {len(train_names)}, Test: {len(test_names)}")
    print(f"  Step: {args.step} (every {args.step}-th view)")

    if any(p != 0 for p in args.pitch) or any(y != 0 for y in args.yaw):
        pitch_str = ", ".join(f"{p}°" for p in args.pitch)
        yaw_str = ", ".join(f"{y}°" for y in args.yaw)
        print(f"  Pitch pattern: [{pitch_str}] (cycling per view)")
        print(f"  Yaw pattern: [{yaw_str}] (cycling per view)")

    # ── 2. Collect and sort valid camera views ──────────────────────────
    valid_views = []
    for frame_entry in ext_data["extrinsics"]:
        cam_key = frame_entry["key"]
        filename = cam_file_dict.get(cam_key)
        if filename is None:
            print(f"  WARNING: No filename found for key {cam_key}, skipping")
            continue

        image_rel = os.path.join("images", filename)
        image_path = src / image_rel
        if not image_path.exists():
            print(f"  WARNING: Image not found: {image_path}, skipping")
            continue

        image_name = Path(filename).stem
        is_train = image_name in train_names
        is_test = image_name in test_names
        if not is_train and not is_test:
            continue

        valid_views.append((image_name, cam_key, frame_entry))

    # Sort by image_name to preserve sequential order (adjacent frames are close)
    valid_views.sort(key=lambda x: x[0])

    # Apply step sampling from the sorted list
    if args.step > 1:
        sampled_views = [valid_views[i] for i in range(0, len(valid_views), args.step)]
        print(f"  Step={args.step}: {len(valid_views)} → {len(sampled_views)} views will be processed")
        valid_views = sampled_views
    else:
        print(f"  All {len(valid_views)} views will be processed")

    # ── 3. Create output structure ──────────────────────────────────────
    out_images_dir = dst / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    # ── 4. Process each camera view ─────────────────────────────────────
    camera_angle_x = np.pi / 2.0
    train_frames = []
    test_frames = []

    white_bg = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    for view_idx, (image_name, cam_key, frame_entry) in enumerate(valid_views):
        is_train = image_name in train_names
        is_test = image_name in test_names
        filename = cam_file_dict[cam_key]

        # OpenMVG: rotation = world→camera, center = camera position in world
        R_w2c = np.array(frame_entry["value"]["rotation"], dtype=np.float64)
        center = np.array(frame_entry["value"]["center"], dtype=np.float64)

        R_c2w = R_w2c.T

        image_rel = os.path.join("images", filename)
        image_path = src / image_rel

        # Load equirectangular image (RGBA → RGB)
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            print(f"  WARNING: Could not load {image_path}, skipping")
            continue

        if img_bgr.shape[2] == 4:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGBA).astype(np.float32)
            alpha = rgb[:, :, 3:4] / 255.0
            rgb = rgb[:, :, :3] / 255.0
            rgb = rgb * alpha + white_bg * (1.0 - alpha)
            rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        else:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)

        print(f"  [{view_idx + 1}/{len(valid_views)}] {image_name} ({rgb.shape[1]}x{rgb.shape[0]})")

        pitch = args.pitch[view_idx % len(args.pitch)]
        yaw = args.yaw[view_idx % len(args.yaw)]
        pitch_rad = np.deg2rad(pitch)
        yaw_rad = np.deg2rad(yaw)
        R_pitch = np.array([[1, 0, 0],
                            [0, np.cos(pitch_rad), np.sin(pitch_rad)],
                            [0, -np.sin(pitch_rad), np.cos(pitch_rad)]], dtype=np.float64)
        R_yaw = np.array([[np.cos(yaw_rad), 0, np.sin(yaw_rad)],
                          [0, 1, 0],
                          [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]], dtype=np.float64)
        R_offset = R_pitch @ R_yaw

        for face_name in selected_faces:
            face_rot = R_offset @ face_rots[face_name]

            # Camera-to-world rotation for this face
            R_face_c2w = R_c2w @ R_offset @ face_rots[face_name]

            # Build Blender-format c2w matrix
            # Blender: Y-up, Z-back
            # Reader will do c2w[:3, 1:3] *= -1 to convert to RTR-GS convention
            c2w_blender = np.eye(4, dtype=np.float64)
            c2w_blender[:3, 0] = R_face_c2w[:, 0]   # right (unchanged after flip)
            c2w_blender[:3, 1] = -R_face_c2w[:, 1]   # Y → negated in reader → becomes R_face[:,1]
            c2w_blender[:3, 2] = -R_face_c2w[:, 2]   # Z → negated in reader → becomes R_face[:,2]
            c2w_blender[:3, 3] = center

            face_img = extract_cube_face(rgb, args.face_size, face_rot)

            face_filename = f"{image_name}_{face_name}.png"
            face_path = out_images_dir / face_filename
            save_face_as_png(face_img, str(face_path))

            frame_entry_out = {
                "file_path": f"./images/{image_name}_{face_name}",
                "transform_matrix": c2w_blender.tolist(),
            }

            if is_train:
                train_frames.append(frame_entry_out)
            if is_test:
                test_frames.append(frame_entry_out)

    # ── 4. Write transforms JSON files ──────────────────────────────────
    if train_frames:
        train_json = {
            "camera_angle_x": camera_angle_x,
            "frames": train_frames,
        }
        with open(dst / "transforms_train.json", "w") as f:
            json.dump(train_json, f, indent=2)
        print(f"Written transforms_train.json with {len(train_frames)} entries")

    if test_frames:
        test_json = {
            "camera_angle_x": camera_angle_x,
            "frames": test_frames,
        }
        with open(dst / "transforms_test.json", "w") as f:
            json.dump(test_json, f, indent=2)
        print(f"Written transforms_test.json with {len(test_frames)} entries")

    # ── 5. Copy and convert point cloud (with normals) ──────────────────
    from plyfile import PlyData, PlyElement

    ply_candidates = ["colorized.ply", "pcd.ply", "scene_dense.ply", "scene_dense_SGM.ply"]
    ply_copied = False
    for ply_name in ply_candidates:
        ply_src = src / ply_name
        if ply_src.exists():
            src_ply = PlyData.read(str(ply_src))
            verts = src_ply["vertex"]
            xyz = np.column_stack([verts["x"], verts["y"], verts["z"]])
            rgb = np.column_stack([verts["red"], verts["green"], verts["blue"]])
            if rgb.dtype == np.float32 or rgb.dtype == np.float64:
                rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

            normals = np.random.randn(*xyz.shape).astype(np.float32)
            normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

            ply_dtype = [
                ("x", "f4"), ("y", "f4"), ("z", "f4"),
                ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ]
            elements = np.empty(xyz.shape[0], dtype=ply_dtype)
            attributes = np.column_stack([xyz, normals, rgb])
            elements[:] = list(map(tuple, attributes))
            PlyData([PlyElement.describe(elements, "vertex")]).write(str(dst / "points3d.ply"))

            print(f"Converted {ply_name} → points3d.ply ({len(xyz)} points)")
            ply_copied = True
            break
    if not ply_copied:
        print("  WARNING: No point cloud found. RTR-GS will generate random points.")

    print(f"\nDone! Converted dataset saved to {dst}")
    print(f"  Total train frames: {len(train_frames)}")
    print(f"  Total test frames: {len(test_frames)}")
    print(f"  Face size: {args.face_size}x{args.face_size}")
    print(f"  Faces: {selected_faces}")


if __name__ == "__main__":
    main()
