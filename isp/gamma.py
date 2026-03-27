"""Apply sRGB gamma encoding to linear RGB values."""

import numpy as np

SRGB_THRESHOLD = 0.0031308
SRGB_LINEAR_SCALE = 12.92
SRGB_POWER_SCALE = 1.055
SRGB_POWER_OFFSET = 0.055
SRGB_GAMMA = 1.0 / 2.4


def apply_gamma(rgb: np.ndarray) -> np.ndarray:
    """Encode linear RGB into sRGB and clip to [0, 1]."""
    encoded = np.where(
        rgb < SRGB_THRESHOLD,
        SRGB_LINEAR_SCALE * rgb,
        SRGB_POWER_SCALE * (rgb ** SRGB_GAMMA) - SRGB_POWER_OFFSET,
    )
    return np.clip(encoded, 0.0, 1.0).astype(np.float32)



if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from isp.black_level import load_raw, subtract_black_level
    from isp.white_balance import white_balance
    from isp.demosaic import demosaic
    from isp.ccm import apply_ccm

    npy_path = os.path.join(os.path.dirname(__file__), "../data/frame.npy")

    print("Running pipeline up to gamma...")
    bayer_raw, meta = load_raw(npy_path)
    bayer = subtract_black_level(bayer_raw, meta["black_level"], meta["white_level"])
    bayer = white_balance(bayer)
    rgb   = demosaic(bayer, pattern=meta["bayer_pattern"])
    rgb   = apply_ccm(rgb)

    print(f"  Before gamma — mean={rgb.mean():.4f}  (linear, looks flat)")
    encoded = apply_gamma(rgb)
    print(f"  After gamma  — mean={encoded.mean():.4f}  (gamma-encoded, should be brighter/higher mean)")

    assert encoded.shape == rgb.shape,      "shape must be unchanged (H, W, 3)"
    assert encoded.dtype == np.float32,     "dtype must be float32"
    assert encoded.min() >= 0.0,            "min must be >= 0.0"
    assert encoded.max() <= 1.0,            "max must be <= 1.0"
    assert encoded.mean() > rgb.mean(),     "gamma must increase mean brightness"

    from PIL import Image
    preview_path = os.path.join(os.path.dirname(__file__), "../data/gamma_preview.jpg")
    Image.fromarray((encoded * 255).astype(np.uint8)).save(preview_path, quality=90)
    print(f"  Preview saved → {preview_path}")
    print("All checks passed.")
