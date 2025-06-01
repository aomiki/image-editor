# Image editor
[![Build with LodePNG](https://github.com/aomiki/image-editor/actions/workflows/c-cpp.yml/badge.svg)](https://github.com/aomiki/image-editor/actions/workflows/c-cpp.yml)
[![Build with CUDA](https://github.com/aomiki/image-editor/actions/workflows/cuda.yml/badge.svg)](https://github.com/aomiki/image-editor/actions/workflows/cuda.yml)

Image editor with the following features:
* Image transformations (crop, reflect, rotate, shear)
* Filters (blur, grayscale)
* Hardware acceleration (CUDA - for transformations, filters, image codec)
* GUI (QT-based)

## Building

The project can be built with different backends for image processing. The workflow has been tested on Arch Linux and would likely run into errors on any other distro / OS.
Every build configuration requires QT6 development libraries and boost program_options library.

### No acceleration (CPU)
This target builds the project using the LodePNG library as image codec, CPU-based implementation for image processing. This requires OpenBLAS library installed.

```bash
make graphics-lode.out
```

### NVIDIA GPU acceleration

This target builds the project leveraging NVIDIA CUDA for GPU acceleration. This requires a compatible NVIDIA GPU and the CUDA Toolkit installed.
```bash
make graphics-cuda.out
```

### Cleaning build files

Clean the generated object files and executables:
```
make clean
```

## Running

Create `output` folder, then run:

### No acceleration:

Get help for CLI:
```bash
./graphics-lode.out --help
```

Launch GUI:
```bash
./graphics-lode.out --gui
```

### NVIDIA GPU acceleration
Get help for CLI:
```bash
./graphics-cuda.out --help
```

Launch GUI:
```bash
./graphics-cuda.out --gui
```
