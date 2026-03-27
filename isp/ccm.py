"""Apply a fixed 3x3 color correction matrix (CCM) to RGB images."""

import json
import numpy as np

CCM_D65 = np.array([
    [1.56879, -0.42159, -0.14719],
    [-0.27275, 1.59354, -0.32079],
    [-0.02862, -0.40662, 1.43525],
], dtype=np.float32)


def load_ccm_table(json_path: str) -> list[tuple[int, np.ndarray]]:
    with open(json_path) as f:
        data = json.load(f)
    for item in data["algorithms"]:
        if "rpi.ccm" in item:
            entries = item["rpi.ccm"]["ccms"]
            table = [
                (e["ct"], np.array(e["ccm"], dtype=np.float32).reshape(3, 3))
                for e in entries
            ]
            return sorted(table, key=lambda x: x[0])
    raise ValueError("rpi.ccm not found in calibration JSON")


def interpolate_ccm(ct: float, table: list[tuple[int, np.ndarray]]) -> np.ndarray:
    if ct <= table[0][0]:
        return table[0][1]
    if ct >= table[-1][0]:
        return table[-1][1]
    for i in range(len(table) - 1):
        ct_lo, ccm_lo = table[i]
        ct_hi, ccm_hi = table[i + 1]
        if ct_lo <= ct <= ct_hi:
            t = (ct - ct_lo) / (ct_hi - ct_lo)
            return ((1 - t) * ccm_lo + t * ccm_hi).astype(np.float32)


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

    json_path = os.path.join(os.path.dirname(__file__), "../imx708.json")
    npy_path  = os.path.join(os.path.dirname(__file__), "../data/frame.npy")

    print("Loading CCM table...")
    table = load_ccm_table(json_path)
    cts = [ct for ct, _ in table]
    print(f"  Loaded {len(table)} entries: {cts}")
    assert len(table) == 5, f"expected 5 CCM entries, got {len(table)}"
    assert cts == sorted(cts), "entries must be sorted by CT"

    ccm_5910 = interpolate_ccm(5910, table)
    assert np.allclose(ccm_5910, CCM_D65, atol=1e-4), \
        f"interpolate_ccm(5910) should match CCM_D65\ngot:\n{ccm_5910}"
    print("  interpolate_ccm(5910) matches CCM_D65 ✓")

    ccm_low  = interpolate_ccm(1000, table)
    ccm_high = interpolate_ccm(9999, table)
    assert np.allclose(ccm_low,  table[0][1]), "should clamp to lowest CT"
    assert np.allclose(ccm_high, table[-1][1]), "should clamp to highest CT"
    print("  boundary clamping ✓")

    ccm_mid = interpolate_ccm(4000, table)
    assert not np.allclose(ccm_mid, table[0][1]) and not np.allclose(ccm_mid, table[1][1]), \
        "mid-point CCM should differ from both endpoints"
    print("  mid-point interpolation ✓")

    if not os.path.exists(npy_path):
        print(f"  Skipping pipeline smoke test — {npy_path} not found")
        print("All checks passed.")
        sys.exit(0)

    print("Running pipeline up to CCM...")
    bayer_raw, meta = load_raw(npy_path)
    bayer = subtract_black_level(bayer_raw, meta["black_level"], meta["white_level"])
    bayer = white_balance(bayer)
    rgb   = demosaic(bayer, pattern=meta["bayer_pattern"])

    print(f"  Before CCM — R_mean={rgb[...,0].mean():.4f}  "
          f"G_mean={rgb[...,1].mean():.4f}  B_mean={rgb[...,2].mean():.4f}")

    ccm = interpolate_ccm(5910, table)
    corrected = apply_ccm(rgb, ccm)
    print(f"  After CCM  — R_mean={corrected[...,0].mean():.4f}  "
          f"G_mean={corrected[...,1].mean():.4f}  B_mean={corrected[...,2].mean():.4f}")

    assert corrected.shape == rgb.shape,  "shape must be unchanged (H, W, 3)"
    assert corrected.dtype == np.float32, "dtype must be float32"
    assert corrected.min() >= 0.0,        "min must be >= 0.0 after clip"
    assert corrected.max() <= 1.0,        "max must be <= 1.0 after clip"

    from PIL import Image
    preview_path = os.path.join(os.path.dirname(__file__), "../data/ccm_preview.jpg")
    Image.fromarray((corrected * 255).astype(np.uint8)).save(preview_path, quality=90)
    print(f"  Preview saved → {preview_path}")
    print("All checks passed.")
