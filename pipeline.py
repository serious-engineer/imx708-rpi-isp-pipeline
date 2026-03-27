"""pipeline.py — Run the full ISP pipeline from raw Bayer to outputs.
Usage:
    source venv/bin/activate
    python pipeline.py --input data/frame.npy --output-dir data/out

"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

from isp.black_level import subtract_black_level
from isp.white_balance import white_balance
from isp.demosaic import demosaic
from isp.ccm import apply_ccm
from isp.gamma import apply_gamma
from isp.yuv import rgb_to_yuv


@dataclass(frozen=True)
class FrameMeta:
    black_level: int
    white_level: int
    bit_depth: int
    bayer_pattern: str
    height: int
    width: int


def load_frame(npy_path: str) -> tuple[np.ndarray, FrameMeta]:
    """Load a raw Bayer frame and its metadata."""
    bayer_raw = np.load(npy_path)
    with open(npy_path + ".json","r") as file1:
        raw_json = json.load(file1)
    meta = FrameMeta(
        black_level=raw_json["black_level"],
        white_level=raw_json["white_level"],
        bit_depth=raw_json["bit_depth"],
        bayer_pattern=raw_json["bayer_pattern"],
        height=raw_json["height"],
        width=raw_json["width"],
    )
    assert bayer_raw.shape == (meta.height, meta.width),    "shape does not match"
    return bayer_raw, meta

def ensure_dir(path: str) -> None:
    """Create output directory if needed."""
    os.makedirs(path, exist_ok=True)


def save_rgb_preview(rgb: np.ndarray, out_path: str) -> None:
    """Save an RGB float32 [0,1] image as JPEG."""
    img = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8))
    img.save(out_path, quality=92)


def save_yuv_visualizations(yuv: np.ndarray, out_dir: str) -> None:
    """Save Y, U, V visualization images.

    - Y is clipped to [0,1] and saved as grayscale.
    - U/V are shifted by +0.5 for visualization, then clipped.
    """
    y = np.clip(yuv[..., 0], 0.0, 1.0)
    u_vis = np.clip(yuv[..., 1] + 0.5, 0.0, 1.0)
    v_vis = np.clip(yuv[..., 2] + 0.5, 0.0, 1.0)

    Image.fromarray((y * 255).astype(np.uint8)).save(os.path.join(out_dir, "y.jpg"), quality=92)
    Image.fromarray((u_vis * 255).astype(np.uint8)).save(os.path.join(out_dir, "u.jpg"), quality=92)
    Image.fromarray((v_vis * 255).astype(np.uint8)).save(os.path.join(out_dir, "v.jpg"), quality=92)


def run_pipeline(bayer_raw: np.ndarray, meta: FrameMeta) -> dict[str, np.ndarray]:
    """Run the ISP chain and return gamma-RGB + YUV outputs."""
    bayer = subtract_black_level(bayer_raw, meta.black_level, meta.white_level)
    bayer = white_balance(bayer)
    rgb   = demosaic(bayer, pattern=meta.bayer_pattern)
    rgb   = apply_ccm(rgb)
    rgb   = apply_gamma(rgb)
    yuv = rgb_to_yuv(rgb)
    return {"rgb": rgb, "yuv": yuv}



def main() -> None:
    parser = argparse.ArgumentParser(description="Run ISP pipeline on a raw Bayer frame")
    parser.add_argument("--input", default="data/frame.npy", help="Path to input .npy raw frame")
    parser.add_argument("--output-dir", default="data/out", help="Directory to write outputs")
    parser.add_argument("--save-yuv", action="store_true", help="Also save Y/U/V visualization images")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    bayer_raw, meta = load_frame(args.input)
    outputs = run_pipeline(bayer_raw, meta)

    rgb = outputs["rgb"]
    save_rgb_preview(rgb, os.path.join(args.output_dir, "rgb_gamma.jpg"))

    if args.save_yuv:
        save_yuv_visualizations(outputs["yuv"], args.output_dir)

    print(f"Wrote: {os.path.join(args.output_dir, 'rgb_gamma.jpg')}")
    if args.save_yuv:
        print(f"Wrote: {os.path.join(args.output_dir, 'y.jpg')}, u.jpg, v.jpg")


if __name__ == "__main__":
    main()
