# imx708-rpi-isp-pipeline

From-scratch IMX708 ISP pipeline on Raspberry Pi: raw Bayer capture, black level correction, white balance, demosaic, CCM, gamma, RGB to YUV, and stage visualization.

## ISP Pipeline (IMX708) - Python

End-to-end ISP prototype for Raspberry Pi Camera Module 3 (IMX708), built step-by-step.
Current status: full Python pipeline implemented and runnable.

## What Is Implemented

| # | Stage | File | Status |
|---|---|---|---|
| 1 | Raw Bayer capture | `capture.py` | ✅ Done |
| 2 | Black level subtraction + normalization | `isp/black_level.py` | ✅ Done |
| 3 | White balance (gray-world) | `isp/white_balance.py` | ✅ Done |
| 4 | Demosaicing | `isp/demosaic.py` | ✅ Done |
| 5 | Color correction matrix (CCM) | `isp/ccm.py` | ✅ Done |
| 6 | sRGB gamma encoding | `isp/gamma.py` | ✅ Done |
| 7 | RGB → YUV (BT.601) | `isp/yuv.py` | ✅ Done |
| 8 | Full pipeline CLI | `pipeline.py` | ✅ Done |
| 9 | Stage visualizer + live capture + rpicam compare | `visualize.py` | ✅ Done |

## Project Layout

```text
isp_pipeline/
  capture.py
  pipeline.py
  visualize.py
  imx708.json
  isp/
    __init__.py
    black_level.py
    white_balance.py
    demosaic.py
    ccm.py
    gamma.py
    yuv.py
  cpp/                  # next phase (C++)
  data/                 # runtime outputs (currently cleaned)
  venv/                 # project virtual environment
```

## Environment

Use the project venv (not `photopipe-venv`):

```bash
cd /home/pi/isp_pipeline
source venv/bin/activate
```

## Sensor / Data Notes

- Sensor: IMX708
- Effective raw format used: `SBGGR16` (10-bit data packed into 16-bit container)
- Bayer pattern used in pipeline: `BGGR`
- Resolution: `4608 x 2592`
- White level used: `65472` (`1023 << 6`)
- Black level currently set in metadata from capture script (project-calibrated)

## Main Commands

### 1) Capture one raw frame

```bash
cd /home/pi/isp_pipeline
source venv/bin/activate
python capture.py --output data/frame.npy
```

Outputs:
- `data/frame.npy`
- `data/frame.npy.json`

### 2) Run the full pipeline

```bash
python pipeline.py --input data/frame.npy --output-dir data/out --save-yuv
```

Outputs:
- `data/out/rgb_gamma.jpg`
- `data/out/y.jpg`
- `data/out/u.jpg`
- `data/out/v.jpg`

### 3) Visualize all stages from existing raw frame

```bash
python visualize.py --input data/frame.npy --save data/stage_grid.png
```

### 4) Live capture + full stage plot

```bash
python visualize.py --capture
```

Default outputs:
- `data/frame_live.npy`
- `data/frame_live.npy.json`
- `data/stage_grid_live.png`

### 5) Live capture + compare against Raspberry Pi ISP

```bash
python visualize.py --capture --compare-rpicam
```

Outputs:
- Pipeline output: `data/pipeline_rgb_live.jpg`
- rpicam output: `data/rpicam_reference.jpg`
- Side-by-side: `data/compare_side_by_side.jpg`
- Stage grid: `data/stage_grid_live.png`

## Known Gaps / Next Improvements

- Mild green tint can appear depending on scene lighting.
- Current CCM is fixed; better color requires CT-based CCM interpolation from `imx708.json`.
- White balance is gray-world only; manual gains / AWB improvements are next.

## Next Phase

C++ port under `cpp/`:
- mirror stage order
- validate output against Python reference
- then integrate real-time path

## References

- Malvar, He, Cutler (ICASSP 2004):
  [High-Quality Linear Interpolation for Demosaicing of Bayer-Patterned Color Images](https://stanford.edu/class/ee367/reading/Demosaicing_ICASSP04.pdf)
- sRGB transfer function reference:
  [sRGB (Wikipedia)](https://en.wikipedia.org/wiki/SRGB)
