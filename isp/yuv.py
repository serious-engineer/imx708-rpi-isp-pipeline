"""Convert gamma-encoded RGB to YUV using BT.601 coefficients."""

import numpy as np

def rgb_to_yuv(rgb: np.ndarray) -> np.ndarray:
    """Return YUV image with channels [Y, U, V]."""
    rgb = rgb.astype(np.float32)

    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.10001 * b
    yuv = np.stack((y, u, v), axis=-1)

    return yuv.astype(np.float32)


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from isp.black_level import load_raw, subtract_black_level
    from isp.white_balance import white_balance
    from isp.demosaic import demosaic
    from isp.ccm import apply_ccm
    from isp.gamma import apply_gamma

    npy_path = os.path.join(os.path.dirname(__file__), "../data/frame.npy")

    print("Running pipeline up to YUV...")
    bayer_raw, meta = load_raw(npy_path)
    bayer = subtract_black_level(bayer_raw, meta["black_level"], meta["white_level"])
    bayer = white_balance(bayer)
    rgb   = demosaic(bayer, pattern=meta["bayer_pattern"])
    rgb   = apply_ccm(rgb)
    rgb   = apply_gamma(rgb)

    yuv = rgb_to_yuv(rgb)
    print(f"  Output — shape={yuv.shape}, dtype={yuv.dtype}")
    print(f"  Y: min={yuv[...,0].min():.4f} max={yuv[...,0].max():.4f} mean={yuv[...,0].mean():.4f}")
    print(f"  U: min={yuv[...,1].min():.4f} max={yuv[...,1].max():.4f} mean={yuv[...,1].mean():.4f}")
    print(f"  V: min={yuv[...,2].min():.4f} max={yuv[...,2].max():.4f} mean={yuv[...,2].mean():.4f}")

    assert yuv.shape == rgb.shape, "shape must be unchanged (H, W, 3)"
    assert yuv.dtype == np.float32, "dtype must be float32"

    from PIL import Image
    out_dir = os.path.join(os.path.dirname(__file__), "../data")

    y = np.clip(yuv[..., 0], 0.0, 1.0)
    Image.fromarray((y * 255).astype(np.uint8)).save(os.path.join(out_dir, "y_channel.jpg"), quality=90)

    u_vis = np.clip(yuv[..., 1] + 0.5, 0.0, 1.0)
    v_vis = np.clip(yuv[..., 2] + 0.5, 0.0, 1.0)
    Image.fromarray((u_vis * 255).astype(np.uint8)).save(os.path.join(out_dir, "u_channel_vis.jpg"), quality=90)
    Image.fromarray((v_vis * 255).astype(np.uint8)).save(os.path.join(out_dir, "v_channel_vis.jpg"), quality=90)

    print("  Saved previews → data/y_channel.jpg, data/u_channel_vis.jpg, data/v_channel_vis.jpg")
    print("All checks passed.")
