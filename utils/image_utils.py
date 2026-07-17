import torch
import numpy as np
import matplotlib


def _get_colormap(name):
    try:
        return matplotlib.colormaps[name]
    except AttributeError:
        import matplotlib.pyplot as plt
        return plt.get_cmap(name)


def colorize_depth(depth, cmap_name="turbo", near=None, far=None, normalized=False,
                   invalid_black=True, curve="log"):
    """Convert a depth tensor/image to a 3-channel color visualization tensor.

    curve:
        "log"    - emphasize near-depth differences (default).
        "linear" - preserve linear normalized depth.
    """
    if isinstance(depth, torch.Tensor):
        device = depth.device
        depth_np = depth.detach().float().cpu().squeeze().numpy()
    else:
        device = None
        depth_np = np.asarray(depth, dtype=np.float32).squeeze()

    if depth_np.ndim == 3:
        if depth_np.shape[0] in (1, 3):
            depth_np = depth_np[0]
        elif depth_np.shape[-1] in (1, 3):
            depth_np = depth_np[..., 0]

    if depth_np.ndim != 2:
        raise ValueError(f"Expected a single depth map, got shape {depth_np.shape}")

    finite_mask = np.isfinite(depth_np)
    if curve not in ("linear", "log"):
        raise ValueError(f"Unsupported depth curve '{curve}'. Expected 'linear' or 'log'.")

    if normalized:
        valid_mask = finite_mask
        depth_norm = np.clip(np.nan_to_num(depth_np, nan=0.0), 0.0, 1.0)
        if curve == "log":
            depth_norm = 1.0 - np.log1p(depth_norm * 255.0) / np.log1p(255.0)
    else:
        valid_mask = finite_mask & (depth_np > 1e-8)
        if not valid_mask.any():
            out = np.zeros((3, depth_np.shape[0], depth_np.shape[1]), dtype=np.float32)
            return torch.from_numpy(out).to(device) if device is not None else torch.from_numpy(out)

        lo = float(near) if near is not None else float(depth_np[valid_mask].min())
        hi = float(far) if far is not None else float(depth_np[valid_mask].max())
        if curve == "log":
            eps = np.finfo(np.float32).eps
            lo = max(lo, eps)
            hi = max(hi, lo + eps)
            curved_depth = -np.log(np.maximum(depth_np, eps))
            curved_lo = -np.log(lo)
            curved_hi = -np.log(hi)
            lower = min(curved_lo, curved_hi)
            denom = max(abs(curved_hi - curved_lo), eps)
            depth_norm = np.clip((curved_depth - lower) / denom, 0.0, 1.0)
        else:
            denom = max(hi - lo, np.finfo(np.float32).eps)
            depth_norm = np.clip((depth_np - lo) / denom, 0.0, 1.0)
        depth_norm = np.nan_to_num(depth_norm, nan=0.0)

    colormap = _get_colormap(cmap_name)
    depth_rgb = colormap(depth_norm)[..., :3].astype(np.float32)
    if invalid_black:
        depth_rgb[~valid_mask] = 0.0

    out = torch.from_numpy(depth_rgb).permute(2, 0, 1).float()
    return out.to(device) if device is not None else out


def visualize_depth(depth, near=0.2, far=13):
    depth = depth[0].detach().cpu().numpy()
    colormap = _get_colormap('turbo')
    curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
    eps = np.finfo(np.float32).eps
    near = near if near else depth.min()
    far = far if far else depth.max()
    near -= eps
    far += eps
    near, far, depth = [curve_fn(x) for x in [near, far, depth]]
    depth = np.nan_to_num(
        np.clip((depth - np.minimum(near, far)) / np.abs(far - near), 0, 1))
    vis = colormap(depth)[:, :, :3]

    out_depth = np.clip(np.nan_to_num(vis), 0., 1.)
    return torch.from_numpy(out_depth).float().cuda().permute(2, 0, 1)


def mse(img1, img2):
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    return 20 * torch.log10(1.0 / torch.sqrt(mse(img1, img2)))

def mae(img1, img2):
    return torch.mean(torch.abs(img1 - img2))
