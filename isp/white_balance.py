"""
isp/white_balance.py — Automatic white balance using the Gray World assumption.
"""

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

if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from isp.black_level import load_raw, subtract_black_level

    npy_path = os.path.join(os.path.dirname(__file__), "../data/frame.npy")

    print("Loading and applying black level...")
    bayer_raw, meta = load_raw(npy_path)
    bayer = subtract_black_level(bayer_raw, meta["black_level"], meta["white_level"])

    print("Computing gray world gains...")
    gain_b, gain_g1, gain_g2, gain_r = compute_gains_gray_world(bayer)
    print(f"  gain_b={gain_b:.4f}  gain_g1={gain_g1:.4f}  "
          f"gain_g2={gain_g2:.4f}  gain_r={gain_r:.4f}")

    print("Applying white balance...")
    balanced = white_balance(bayer)
    print(f"  Output — shape={balanced.shape}, dtype={balanced.dtype}, "
          f"min={balanced.min():.4f}, max={balanced.max():.4f}")

    assert balanced.dtype == np.float32,        "dtype must be float32"
    assert balanced.shape == bayer.shape,       "shape must be unchanged"
    assert balanced.min() >= 0.0,              "min must be >= 0.0"
    assert balanced.max() <= 1.0,              "max must be <= 1.0"

    b_mean  = balanced[0::2, 0::2].mean()
    g1_mean = balanced[0::2, 1::2].mean()
    g2_mean = balanced[1::2, 0::2].mean()
    r_mean  = balanced[1::2, 1::2].mean()
    print(f"  Channel means after WB: B={b_mean:.4f} G1={g1_mean:.4f} "
          f"G2={g2_mean:.4f} R={r_mean:.4f}")
    print("  (These should be approximately equal — gray world satisfied)")
    print("All checks passed.")
