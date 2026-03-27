"""Visualize ISP stages from raw Bayer to YUV."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from isp.black_level import subtract_black_level
from isp.white_balance import white_balance
from isp.demosaic import demosaic
from isp.ccm import apply_ccm, load_ccm_table, interpolate_ccm
from isp.gamma import apply_gamma
from isp.yuv import rgb_to_yuv


def capture_raw_and_meta(output_npy_path: str) -> tuple[np.ndarray, dict]:
    """Capture a fresh frame and load metadata."""
    from capture import create_camera, capture_raw, save_frame

    out_dir = os.path.dirname(output_npy_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cam = create_camera()
    try:
        raw = capture_raw(cam)
        save_frame(raw, output_npy_path)
    finally:
        if hasattr(cam, "close"):
            cam.close()

    time.sleep(0.35)

    with open(output_npy_path + ".json", "r") as f:
        meta = json.load(f)
    return raw, meta


def load_raw_and_meta(npy_path: str) -> tuple[np.ndarray, dict]:
    """Load raw Bayer frame and metadata JSON."""
    raw = np.load(npy_path)
    with open(npy_path + ".json", "r") as f:
        meta = json.load(f)
    return raw, meta


def run_all_stages(raw: np.ndarray, meta: dict) -> dict[str, np.ndarray]:
    """Run the full ISP chain and return all intermediate stages."""
    json_path = os.path.join(os.path.dirname(__file__), "imx708.json")
    ccm_table = load_ccm_table(json_path)

    bayer_bl = subtract_black_level(raw, meta["black_level"], meta["white_level"])
    bayer_wb = white_balance(bayer_bl)
    rgb_dem = demosaic(bayer_wb, pattern=meta["bayer_pattern"])
    ccm = interpolate_ccm(5910, ccm_table)
    rgb_ccm = apply_ccm(rgb_dem, ccm)
    rgb_gamma = apply_gamma(rgb_ccm)
    yuv = rgb_to_yuv(rgb_gamma)

    return {
        "raw": raw,
        "black_level": bayer_bl,
        "white_balance": bayer_wb,
        "demosaic": rgb_dem,
        "ccm": rgb_ccm,
        "gamma": rgb_gamma,
        "yuv": yuv,
    }


def _to_uint8_gray(x: np.ndarray) -> np.ndarray:
    """Map float/int array to displayable uint8 grayscale."""
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32)
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return (np.clip(x, 0.0, 1.0) * 255).astype(np.uint8)


def plot_stages(stages: dict[str, np.ndarray], save_path: str | None = None) -> None:
    """Plot ISP stages in a 2x3 grid."""
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs[0, 0].imshow(_to_uint8_gray(stages["raw"]), cmap="gray", vmin=0, vmax=255)
    axs[0, 0].set_title("Raw_Bayer")

    axs[0, 1].imshow(_to_uint8_gray(stages["black_level"]), cmap="gray", vmin=0, vmax=255)
    axs[0, 1].set_title("After_BlackLevel")

    axs[0, 2].imshow(_to_uint8_gray(stages["white_balance"]), cmap="gray", vmin=0, vmax=255)
    axs[0, 2].set_title("After_WhiteBalance")

    axs[1, 0].imshow(np.clip(stages["demosaic"], 0.0, 1.0))
    axs[1, 0].set_title("Demosaic_RGB")

    axs[1, 1].imshow(np.clip(stages["ccm"], 0.0, 1.0))
    axs[1, 1].set_title("After_CCM")

    axs[1, 2].imshow(np.clip(stages["gamma"], 0.0, 1.0))
    axs[1, 2].set_title("After_Gamma_sRGB")

    for ax in axs.ravel():
        ax.axis("off")

    fig.tight_layout()

    if save_path:
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved stage grid → {save_path}")

    plt.show()


def save_rgb_preview(rgb: np.ndarray, out_path: str) -> None:
    """Save gamma RGB output as JPEG."""
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    img = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8))
    img.save(out_path, quality=92)
    print(f"Saved pipeline RGB → {out_path}")


def capture_rpicam_jpeg(out_path: str, width: int, height: int) -> None:
    """Capture reference JPEG using Raspberry Pi ISP (rpicam-still)."""
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "rpicam-still",
        "-o",
        out_path,
        "--width",
        str(width),
        "--height",
        str(height),
        "--nopreview",
    ]
    last_err = None
    for _ in range(3):
        try:
            subprocess.run(cmd, check=True)
            print(f"Saved rpicam reference → {out_path}")
            return
        except subprocess.CalledProcessError as err:
            last_err = err
            time.sleep(0.5)

    raise RuntimeError(
        "rpicam-still could not acquire the camera after 3 attempts. "
        "Close any camera-using process and retry."
    ) from last_err


def save_side_by_side(left_path: str, right_path: str, out_path: str) -> None:
    """Create side-by-side comparison image from two files."""
    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")

    w = min(left.width, right.width)
    h = min(left.height, right.height)
    left = left.resize((w, h))
    right = right.resize((w, h))

    canvas = Image.new("RGB", (w * 2, h + 50), (20, 20, 20))
    canvas.paste(left, (0, 50))
    canvas.paste(right, (w, 50))

    draw = ImageDraw.Draw(canvas)
    draw.text((20, 15), "Our pipeline", fill=(255, 255, 255))
    draw.text((w + 20, 15), "rpicam ISP", fill=(255, 255, 255))

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    canvas.save(out_path, quality=95)
    print(f"Saved side-by-side comparison → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize all ISP pipeline stages")
    parser.add_argument(
        "--input",
        default="data/frame.npy",
        help="Path to raw .npy frame (ignored if --capture is set)",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Capture a fresh frame from the camera before plotting",
    )
    parser.add_argument(
        "--capture-output",
        default="data/frame_live.npy",
        help="Where to save the captured raw frame (default: data/frame_live.npy)",
    )
    parser.add_argument("--save", default=None, help="Optional output path for stage grid PNG")
    parser.add_argument(
        "--save-pipeline-rgb",
        default=None,
        help="Optional output path for pipeline RGB JPEG (after gamma)",
    )
    parser.add_argument(
        "--compare-rpicam",
        action="store_true",
        help="Also capture an rpicam JPEG and build side-by-side comparison",
    )
    parser.add_argument(
        "--rpicam-output",
        default="data/rpicam_reference.jpg",
        help="Path for rpicam JPEG when --compare-rpicam is used",
    )
    parser.add_argument(
        "--compare-output",
        default="data/compare_side_by_side.jpg",
        help="Path for side-by-side comparison output",
    )
    args = parser.parse_args()

    if args.capture:
        args.compare_rpicam = True
        raw, meta = capture_raw_and_meta(args.capture_output)
        if args.save is None:
            args.save = os.path.join(os.path.dirname(args.capture_output) or ".", "stage_grid_live.png")
        if args.save_pipeline_rgb is None:
            args.save_pipeline_rgb = os.path.join(os.path.dirname(args.capture_output) or ".", "pipeline_rgb_live.jpg")
    else:
        raw, meta = load_raw_and_meta(args.input)

    stages = run_all_stages(raw, meta)

    if args.save_pipeline_rgb:
        save_rgb_preview(stages["gamma"], args.save_pipeline_rgb)

    if args.compare_rpicam:
        if not args.capture:
            raise ValueError("--compare-rpicam requires --capture so both images come from the same scene.")
        capture_rpicam_jpeg(args.rpicam_output, meta["width"], meta["height"])
        if args.save_pipeline_rgb is None:
            args.save_pipeline_rgb = "data/pipeline_rgb_live.jpg"
            save_rgb_preview(stages["gamma"], args.save_pipeline_rgb)
        save_side_by_side(args.save_pipeline_rgb, args.rpicam_output, args.compare_output)

    plot_stages(stages, save_path=args.save)


if __name__ == "__main__":
    main()
