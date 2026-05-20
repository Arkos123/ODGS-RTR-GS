
import re
import pyexr
import numpy as np
import imageio as imageio
from utils.graphics_utils import rgb_to_srgb

def load_pfm(file: str):
    color = None
    width = None
    height = None
    scale = None
    endian = None
    with open(file, 'rb') as f:
        header = f.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
        dim_match = re.match(br'^(\d+)\s(\d+)\s$', f.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')
        scale = float(f.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = data[::-1, ...]  # cv2.flip(data, 0)

    return np.ascontiguousarray(data)

def load_img_rgb(path):
    
    if path.endswith(".exr"):
        exr_file = pyexr.open(path)
        img = exr_file.get()
        img[..., 0:3] = rgb_to_srgb(img[..., 0:3], clip=False)
    else:
        img = imageio.imread(path)
        img = img / 255
    return img

def get_img_size(path):
    """
    获取图片尺寸（宽,高），不加载像素数据。
    支持常见格式（JPEG, PNG, GIF, BMP, TIFF, EXR等）。
    """
    try:
        # 处理 OpenEXR 格式
        if path.lower().endswith('.exr'):
            import pyexr
            exr_file = pyexr.open(path)
            # 从 header 中提取显示窗口或数据窗口的尺寸
            header = exr_file.header
            # 优先使用 dataWindow（实际像素数据区域）
            dw = header.get('dataWindow')
            if dw is None:
                dw = header.get('displayWindow')  # 备用显示窗口
            if dw:
                (xmin, ymin), (xmax, ymax) = dw
                width = xmax - xmin + 1
                height = ymax - ymin + 1
                return width, height
            else:
                raise ValueError("无法从EXR文件中获取尺寸信息")
        else:
            # 其他格式使用 PIL 的懒加载模式
            with Image.open(path) as img:
                width, height = img.size
                return width, height
    except Exception as e:
        # 如果 PIL 无法处理（如某些特殊格式），回退到 imageio 元数据读取（v3）
        try:
            import imageio.v3 as iio
            meta = iio.immeta(path, exclude_applied=False)
            width, height = meta.get('width'), meta.get('height')
            if width is not None and height is not None:
                return width, height
        except:
            pass
        # 最终回退：完整读取图像（可能较慢但确保能获取尺寸）
        import imageio.v3 as iio
        img = iio.imread(path)
        return img.shape[1], img.shape[0]

def load_mask_bool(mask_file):
    mask = imageio.imread(mask_file, mode='L')
    mask = mask.astype(np.float32)
    mask[mask > 0.5] = 1.0

    return mask

def load_depth(tiff_file):
    return imageio.imread(tiff_file, mode='L')

def save_render_orb(file_path_wo_ext, data):
    exr_file = file_path_wo_ext + ".exr"
    pyexr.write(exr_file, data)
    
    png_file = file_path_wo_ext + ".png"
    data = rgb_to_srgb(data) * 255
    imageio.imwrite(png_file, data.astype(np.uint8))

def save_depth_orb(file_path_wo_ext, data):
    data = data[..., 0]
    
    exr_file = file_path_wo_ext + ".exr"
    pyexr.write(exr_file, data)
    
    png_file = file_path_wo_ext + ".png"
    
    mask = data != 0
    data[mask] = (data[mask] - np.min(data[mask])) / (np.max(data[mask])- np.min(data[mask]))
    data = data * 255
    imageio.imwrite(png_file, data.astype(np.uint8))

def save_normal_orb(file_path_wo_ext, data):
    exr_file = file_path_wo_ext + ".exr"
    pyexr.write(exr_file, data)
    
    png_file = file_path_wo_ext + ".png"
    
    data = data * 0.5 + 0.5
    data = data * 255
    imageio.imwrite(png_file, data.astype(np.uint8))

def save_albedo_orb(file_path_wo_ext, data):
    exr_file = file_path_wo_ext + ".exr"
    pyexr.write(exr_file, data)
    
    png_file = file_path_wo_ext + ".png"
    data = np.clip(data, 0.0, 1.0) * 255
    imageio.imwrite(png_file, data.astype(np.uint8))

def save_roughness_orb(file_path_wo_ext, data):
    data = data[..., 0]

    exr_file = file_path_wo_ext + ".exr"
    pyexr.write(exr_file, data)
    
    png_file = file_path_wo_ext + ".png"
    data = np.clip(data, 0.0, 1.0) * 255
    imageio.imwrite(png_file, data.astype(np.uint8))
