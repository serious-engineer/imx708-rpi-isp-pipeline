"""Apply a fixed 3x3 color correction matrix (CCM) to RGB images."""

import numpy as np

CCM_D65 = np.array([
    [1.56879, -0.42159, -0.14719],
    [-0.27275, 1.59354, -0.32079],
    [-0.02862, -0.40662, 1.43525],
], dtype=np.float32)


def apply_ccm(rgb: np.ndarray, ccm: np.ndarray = CCM_D65) -> np.ndarray:
    """Apply CCM and clip result to [0, 1]."""
    pixels = rgb.reshape(-1, 3)
    corrected_pixels = pixels @ ccm.T
    corrected = corrected_pixels.reshape(rgb.shape)
    corrected = np.clip(corrected, 0.0, 1.0)
    return corrected.astype(np.float32)


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from isp.black_level import load_raw, subtract_black_level
    from isp.white_balance import white_balance
    from isp.demosaic import demosaic

    npy_path = os.path.join(os.path.dirname(__file__), "../data/frame.npy")

    print("Running pipeline up to CCM...")
    bayer_raw, meta = load_raw(npy_path)
    bayer = subtract_black_level(bayer_raw, meta["black_level"], meta["white_level"])
    bayer = white_balance(bayer)
    rgb   = demosaic(bayer, pattern=meta["bayer_pattern"])

    print(f"  Before CCM — R_mean={rgb[...,0].mean():.4f}  "
          f"G_mean={rgb[...,1].mean():.4f}  B_mean={rgb[...,2].mean():.4f}")

    corrected = apply_ccm(rgb)
    print(f"  After CCM  — R_mean={corrected[...,0].mean():.4f}  "
          f"G_mean={corrected[...,1].mean():.4f}  B_mean={corrected[...,2].mean():.4f}")

    assert corrected.shape == rgb.shape,    "shape must be unchanged (H, W, 3)"
    assert corrected.dtype == np.float32,   "dtype must be float32"
    assert corrected.min() >= 0.0,          "min must be >= 0.0 after clip"
    assert corrected.max() <= 1.0,          "max must be <= 1.0 after clip"

    from PIL import Image
    preview_path = os.path.join(os.path.dirname(__file__), "../data/ccm_preview.jpg")
    Image.fromarray((corrected * 255).astype(np.uint8)).save(preview_path, quality=90)
    print(f"  Preview saved → {preview_path}")
    print("All checks passed.")
