import json
import numpy as np


def compute_gains_gray_world(bayer: np.ndarray) -> tuple[float, float, float, float]:
    b_channel  = bayer[0::2, 0::2]
    g1_channel = bayer[0::2, 1::2]
    g2_channel = bayer[1::2, 0::2]
    r_channel  = bayer[1::2, 1::2]

    b_channel_mean = np.mean(b_channel)
    g1_channel_mean = np.mean(g1_channel)
    g2_channel_mean = np.mean(g2_channel)
    r_channel_mean = np.mean(r_channel)

    global_mean = (b_channel_mean + g1_channel_mean + g2_channel_mean + r_channel_mean) / 4.0

    gain_b = global_mean / b_channel_mean
    gain_g1 = global_mean / g1_channel_mean
    gain_g2 = global_mean / g2_channel_mean
    gain_r = global_mean / r_channel_mean

    return (gain_b, gain_g1, gain_g2, gain_r)

def apply_gains(
    bayer: np.ndarray,
    gain_b: float,
    gain_g1: float,
    gain_g2: float,
    gain_r: float,
) -> np.ndarray:
    out = bayer.copy()
    out[0::2, 0::2] *= gain_b
    out[0::2, 1::2] *= gain_g1
    out[1::2, 0::2] *= gain_g2
    out[1::2, 1::2] *= gain_r

    out = np.clip(out,0.0,1.0)
    return out.astype(np.float32)

def white_balance(bayer: np.ndarray) -> np.ndarray:
    gain_b, gain_g1, gain_g2, gain_r = compute_gains_gray_world(bayer)
    return apply_gains(bayer, gain_b, gain_g1, gain_g2, gain_r)


def load_ct_locus(json_path: str) -> list[tuple[float, float, float]]:
    with open(json_path) as f:
        data = json.load(f)
    for item in data["algorithms"]:
        if "rpi.awb" in item:
            curve = item["rpi.awb"]["ct_curve"]
            return [(curve[i], curve[i + 1], curve[i + 2]) for i in range(0, len(curve), 3)]
    raise ValueError("rpi.awb not found in calibration JSON")


def _project_to_locus(
    rg: float, bg: float, locus: list[tuple[float, float, float]]
) -> tuple[float, float, float]:
    best_ct, best_rg, best_bg = locus[0]
    best_dist = float("inf")
    for i in range(len(locus) - 1):
        ct0, rg0, bg0 = locus[i]
        ct1, rg1, bg1 = locus[i + 1]
        drg, dbg = rg1 - rg0, bg1 - bg0
        seg_sq = drg * drg + dbg * dbg
        t = max(0.0, min(1.0, ((rg - rg0) * drg + (bg - bg0) * dbg) / seg_sq)) if seg_sq else 0.0
        p_rg = rg0 + t * drg
        p_bg = bg0 + t * dbg
        dist = (rg - p_rg) ** 2 + (bg - p_bg) ** 2
        if dist < best_dist:
            best_dist = dist
            best_ct = ct0 + t * (ct1 - ct0)
            best_rg = p_rg
            best_bg = p_bg
    return best_ct, best_rg, best_bg


def estimate_ct(rg: float, bg: float, locus: list[tuple[float, float, float]]) -> float:
    ct, _, _ = _project_to_locus(rg, bg, locus)
    return ct


def white_balance_locus(
    bayer: np.ndarray, locus: list[tuple[float, float, float]]
) -> tuple[np.ndarray, float]:
    g_mean = (bayer[0::2, 1::2].mean() + bayer[1::2, 0::2].mean()) / 2.0
    rg = bayer[1::2, 1::2].mean() / g_mean
    bg = bayer[0::2, 0::2].mean() / g_mean
    ct, _, _ = _project_to_locus(rg, bg, locus)
    gain_r = 1.0 / rg
    gain_b = 1.0 / bg
    mean_gain = (gain_r + 1.0 + 1.0 + gain_b) / 4.0
    gain_r /= mean_gain
    gain_g = 1.0 / mean_gain
    gain_b /= mean_gain
    return apply_gains(bayer, gain_b, gain_g, gain_g, gain_r), ct

if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from isp.black_level import load_raw, subtract_black_level

    json_path = os.path.join(os.path.dirname(__file__), "../imx708.json")
    npy_path  = os.path.join(os.path.dirname(__file__), "../data/frame.npy")

    print("Loading CT locus...")
    locus = load_ct_locus(json_path)
    cts = [ct for ct, _, _ in locus]
    print(f"  Loaded {len(locus)} entries: {cts}")
    assert len(locus) == 5, f"expected 5 locus entries, got {len(locus)}"
    assert cts == sorted(cts), "locus must be sorted by CT"

    for ct_ref, rg_ref, bg_ref in locus:
        ct_est = estimate_ct(rg_ref, bg_ref, locus)
        assert abs(ct_est - ct_ref) < 1.0, \
            f"estimate_ct at exact locus point {ct_ref}K returned {ct_est:.1f}K"
    print("  estimate_ct at exact locus points ✓")

    ct_warm = estimate_ct(0.9, 0.2, locus)
    ct_cool = estimate_ct(0.2, 0.9, locus)
    assert ct_warm <= locus[1][0], f"warm point should project near low CT end, got {ct_warm:.0f}K"
    assert ct_cool >= locus[-2][0], f"cool point should project near high CT end, got {ct_cool:.0f}K"
    print(f"  warm scene → {ct_warm:.0f}K, cool scene → {ct_cool:.0f}K ✓")

    if not os.path.exists(npy_path):
        print(f"  Skipping pipeline smoke test — {npy_path} not found")
        print("All checks passed.")
        sys.exit(0)

    print("Loading and applying black level...")
    bayer_raw, meta = load_raw(npy_path)
    bayer = subtract_black_level(bayer_raw, meta["black_level"], meta["white_level"])

    print("Applying locus-based white balance...")
    balanced, ct = white_balance_locus(bayer, locus)
    print(f"  Estimated CT: {ct:.0f}K")
    print(f"  Output — shape={balanced.shape}, dtype={balanced.dtype}, "
          f"min={balanced.min():.4f}, max={balanced.max():.4f}")

    assert balanced.dtype == np.float32,  "dtype must be float32"
    assert balanced.shape == bayer.shape, "shape must be unchanged"
    assert balanced.min() >= 0.0,         "min must be >= 0.0"
    assert balanced.max() <= 1.0,         "max must be <= 1.0"
    assert 2000 <= ct <= 9000,            f"CT {ct:.0f}K is outside plausible range"

    print("All checks passed.")
