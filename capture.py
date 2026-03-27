"""
capture.py — Capture a raw Bayer frame from the IMX708 and save to disk.

The IMX708 outputs a 10-bit RGGB Bayer mosaic over MIPI CSI-2.
picamera2 unpacks it into a uint16 numpy array (values 0–1023).

Saved files:
    data/frame.npy   — raw Bayer array, shape (H, W), dtype uint16
    data/meta.json   — capture metadata needed by pipeline stages

Usage:
    python capture.py
    python capture.py --output data/my_frame.npy
"""

import argparse
import json
import numpy as np
import picamera2

SENSOR_WIDTH  = 4608
SENSOR_HEIGHT = 2592
BLACK_LEVEL   = 4450
WHITE_LEVEL   = 65472
BAYER_PATTERN = "BGGR"
BIT_DEPTH     = 10


def create_camera():
    """
    Initialise and configure the IMX708 for raw still capture.
    Returns a configured (but not yet started) Picamera2 instance.
    """
    cam = picamera2.Picamera2()
    config =cam.create_still_configuration(raw={"format": "SBGGR16"})
    cam.configure(config)
    return cam


def capture_raw(cam) -> np.ndarray:
    """
    Start the camera, capture one raw frame, stop the camera.
    Returns the raw Bayer array.
    """
    cam.start()
    bayer = cam.capture_array("raw")
    bayer = bayer.view(np.uint16)
    cam.stop()
    return bayer

def save_frame(bayer: np.ndarray, output_path: str) -> None:
    np.save(output_path, bayer)
    metadata = {
        "black_level": BLACK_LEVEL,
        "white_level": WHITE_LEVEL,
        "bit_depth": BIT_DEPTH,
        "bayer_pattern": BAYER_PATTERN,
        "height": bayer.shape[0],
        "width": bayer.shape[1]
    }
    with open(output_path + ".json", "w") as f:
        json.dump(metadata, f)


def main():
    parser = argparse.ArgumentParser(description="Capture a raw Bayer frame from IMX708")
    parser.add_argument("--output", default="data/frame.npy",
                        help="Path to save the raw .npy file (default: data/frame.npy)")
    args = parser.parse_args()

    print(f"Initialising camera...")
    cam = create_camera()

    print("Capturing raw frame...")
    bayer = capture_raw(cam)
    print(f"Captured: shape={bayer.shape}, dtype={bayer.dtype}, "
          f"min={bayer.min()}, max={bayer.max()}")

    save_frame(bayer, args.output)
    print(f"Saved raw frame → {args.output}")
    print(f"Saved metadata  → {args.output}.json")


if __name__ == "__main__":
    main()
