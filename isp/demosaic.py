"""
isp/demosaic.py — Bayer demosaicing: mosaic → RGB image.
"""

import numpy as np
from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004


def demosaic(bayer: np.ndarray, pattern: str = "BGGR") -> np.ndarray:
    rgb = demosaicing_CFA_Bayer_Malvar2004(bayer, pattern)
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb.astype(np.float32)


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from isp.black_level import load_raw, subtract_black_level
    from isp.white_balance import white_balance

    npy_path = os.path.join(os.path.dirname(__file__), "../data/frame.npy")

    print("Loading pipeline stages...")
    bayer_raw, meta = load_raw(npy_path)
    bayer = subtract_black_level(bayer_raw, meta["black_level"], meta["white_level"])
    bayer = white_balance(bayer)

    print("Demosaicing...")
    rgb = demosaic(bayer, pattern=meta["bayer_pattern"])
    print(f"  Output — shape={rgb.shape}, dtype={rgb.dtype}, "
          f"min={rgb.min():.4f}, max={rgb.max():.4f}")

    # Sanity checks
    assert rgb.ndim == 3,                       "output must be 3D (H, W, 3)"
    assert rgb.shape[2] == 3,                   "last dimension must be 3 (RGB)"
    assert rgb.shape[:2] == bayer.shape[:2],    "H, W must be unchanged"
    assert rgb.dtype == np.float32,             "dtype must be float32"
    assert rgb.min() >= 0.0,                    "min must be >= 0.0"
    assert rgb.max() <= 1.0,                    "max must be <= 1.0"

    # Save a quick preview to verify visually
    from PIL import Image
    preview_path = os.path.join(os.path.dirname(__file__), "../data/demosaic_preview.jpg")
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    img.save(preview_path, quality=90)
    print(f"  Preview saved → {preview_path}")
    print("All checks passed.")
