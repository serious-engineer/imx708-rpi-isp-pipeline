"""
isp/black_level.py — Black level subtraction and normalization.
"""

import numpy as np
import os
import json


def subtract_black_level(
    bayer: np.ndarray,
    black_level: int,
    white_level: int,
) -> np.ndarray:
    bayer = bayer.astype(np.float32)
    normalized = (bayer - black_level) / (white_level - black_level)
    normalized = np.clip(normalized,0.0,1.0)
    return normalized.astype(np.float32)


def load_raw(npy_path: str) -> tuple[np.ndarray, dict]:
    file = np.load(npy_path)
    with open(npy_path + ".json", "r") as f:
        meta = json.load(f)
    return file, meta


if __name__ == "__main__":
    npy_path = os.path.join(os.path.dirname(__file__), "../data/frame.npy")

    print("Loading raw frame...")
    bayer, meta = load_raw(npy_path)
    print(f"  Raw    — shape={bayer.shape}, dtype={bayer.dtype}, "
          f"min={bayer.min()}, max={bayer.max()}")

    print("Applying black level subtraction...")
    normalized = subtract_black_level(
        bayer,
        black_level=meta["black_level"],
        white_level=meta["white_level"],
    )
    print(f"  Output — shape={normalized.shape}, dtype={normalized.dtype}, "
          f"min={normalized.min():.4f}, max={normalized.max():.4f}")

    assert normalized.dtype == np.float32,    "dtype must be float32"
    assert normalized.min() >= 0.0,           "min must be >= 0.0 after clip"
    assert normalized.max() <= 1.0,           "max must be <= 1.0 after clip"
    assert normalized.shape == bayer.shape,   "shape must be unchanged"
    print("All checks passed.")
