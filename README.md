# Image editor
[![Build with LodePNG](https://github.com/aomiki/image-editor/actions/workflows/c-cpp.yml/badge.svg)](https://github.com/aomiki/image-editor/actions/workflows/c-cpp.yml)
[![Build with CUDA](https://github.com/aomiki/image-editor/actions/workflows/cuda.yml/badge.svg)](https://github.com/aomiki/image-editor/actions/workflows/cuda.yml)

## Build

* Build with CUDA codec:

    `make graphics-cuda.out`

* Build with LodePNG codec:

    `make graphics-lode.out`

* Build with OpenCL codec:

    `make graphics-opencl.out`

* Clean build files:

    `make clean`

* Clean output folder:

    `make clean-output`

## Run

Create `output` folder, then run:

* If built with CUDA codec:

    `./graphics-cuda.out [options]`

* If built with LodePNG codec:

    `./graphics-lode.out [options]`

* If built with OpenCL codec:

    `./graphics-opencl.out [options]`

## Command Line Options

The program supports the following commands and options:

### Image Crop

```
./graphics-opencl.out --crop <image_file> [--crop_left <pixels>] [--crop_top <pixels>] [--crop_right <pixels>] [--crop_bottom <pixels>]
```

Parameters:
- `--crop_left`: Pixels to crop from left side (default: 200)
- `--crop_top`: Pixels to crop from top side (default: 200)
- `--crop_right`: Pixels to crop from right side (default: 200)
- `--crop_bottom`: Pixels to crop from bottom side (default: 200)

Example:
```
./graphics-opencl.out --crop floyd.png --crop_left 100 --crop_top 150 --crop_right 300 --crop_bottom 250
```

### Image Rotation

```
./graphics-opencl.out --rotate <image_file> [--rotate_angle <degrees>]
```

Parameters:
- `--rotate_angle`: Rotation angle in degrees (default: 90)
  - Only supports 90, 180, and 270 degree rotations

Example:
```
./graphics-opencl.out --rotate floyd.png --rotate_angle 270
```

### Other Commands

* Show help:
```
./graphics-opencl.out --help
```

* Draw border:
```
./graphics-opencl.out --draw_border <image_file>
```

## Notes

- Input images should be placed in the `input` directory
- Processed images will be saved in the `output` directory
- OpenCL version uses GPU acceleration when available, with CPU fallback
- CUDA version requires NVIDIA GPU and runs operations on the GPU
- LodePNG version runs operations on the CPU only

## Performance Comparison

Different implementations offer varying performance characteristics:

- **LodePNG**: CPU-only implementation, best for systems without dedicated graphics hardware
- **OpenCL**: Cross-platform GPU acceleration with CPU fallback, works on AMD, Intel, and NVIDIA hardware
- **CUDA**: NVIDIA-only GPU acceleration, typically offers best performance on supported hardware

For large images, GPU-accelerated versions (OpenCL and CUDA) can provide significant performance improvements over the CPU-only version.
