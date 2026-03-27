# ISP Pipeline — C++ Standalone CLI

A C++ port of the Python ISP prototype. Processes raw `.npy` Bayer frames
through the full pipeline and outputs a `.ppm` image.
Validated against the Python reference implementation (max error ≤ 1 DN).

> **Note:** Complete the Python prototype first. The C++ port begins once
> all Python stages are validated.

## Pipeline Stages

| # | Stage | File | Status |
|---|-------|------|--------|
| 1 | Black level subtraction | `src/isp/BlackLevel.h/cpp` | ⏳ Pending |
| 2 | White balance | `src/isp/WhiteBalance.h/cpp` | ⏳ Pending |
| 3 | Demosaicing | `src/isp/Demosaic.h/cpp` | ⏳ Pending |
| 4 | Color Correction Matrix | `src/isp/CCM.h/cpp` | ⏳ Pending |
| 5 | Gamma correction | `src/isp/Gamma.h/cpp` | ⏳ Pending |
| 6 | RGB → YUV | `src/isp/YUV.h/cpp` | ⏳ Pending |
| 7 | Pipeline runner | `src/Pipeline.h/cpp` | ⏳ Pending |
| 8 | File I/O utilities | `src/Utils.h/cpp` | ⏳ Pending |
| 9 | CLI entry point | `src/main.cpp` | ⏳ Pending |

## Project Structure

```
cpp/
    CMakeLists.txt
    src/
        main.cpp            # CLI: parse args, run pipeline, write output
        Pipeline.h/cpp      # Owns and chains all ISP stages
        Utils.h/cpp         # .npy reader, .ppm writer
        isp/
            BlackLevel.h/cpp
            WhiteBalance.h/cpp
            Demosaic.h/cpp  # Bilinear demosaic
            CCM.h/cpp
            Gamma.h/cpp     # LUT-based sRGB gamma
            YUV.h/cpp       # BT.601 matrix multiply
```

## Build

```bash
cd /home/pi/isp_pipeline/cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

## Usage

```bash
./isp_pipeline --input ../../data/frame.npy --width 4608 --height 2592 --output result.ppm
```

## Validation Against Python

After building, validate the C++ output matches the Python reference:

```bash
# Run Python pipeline first to produce reference output
cd /home/pi/isp_pipeline
/home/pi/photopipe-venv/bin/python pipeline.py --input data/frame.npy --output data/py_out.tiff

# Run C++ pipeline
cd cpp/build
./isp_pipeline --input ../../data/frame.npy --width 4608 --height 2592 --output ../../data/cpp_out.ppm

# Diff them in Python
/home/pi/photopipe-venv/bin/python -c "
import numpy as np
from PIL import Image
py  = np.array(Image.open('data/py_out.tiff')).astype(float)
cpp = np.array(Image.open('data/cpp_out.ppm')).astype(float)
diff = np.abs(py - cpp)
print(f'Max error: {diff.max():.1f} DN')
print(f'Mean error: {diff.mean():.4f} DN')
print('PASS' if diff.max() <= 1.0 else 'FAIL')
"
```

## Dependencies

- CMake ≥ 3.16
- g++ with C++17 support (already on Pi OS)
- No external libraries — stdlib only

## Key Differences from Python

| Aspect | Python | C++ |
|--------|--------|-----|
| Demosaic | Malvar2004 (skimage) | Bilinear (manual) |
| Gamma | `np.where` piecewise | 256-entry float LUT |
| Image I/O | Pillow / tifffile | Custom .ppm writer |
| Performance | ~seconds per frame | ~milliseconds per frame |
