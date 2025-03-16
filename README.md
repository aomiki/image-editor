# Image editor
[![Build with LodePNG](https://github.com/aomiki/image-editor/actions/workflows/c-cpp.yml/badge.svg)](https://github.com/aomiki/image-editor/actions/workflows/c-cpp.yml)
[![Build with CUDA](https://github.com/aomiki/image-editor/actions/workflows/cuda.yml/badge.svg)](https://github.com/aomiki/image-editor/actions/workflows/cuda.yml)

## Build

* Build with CUDA codec:

    `make graphics-cuda.out`

* Build with LodePNG codec:

    `make graphics-lode.out`

* Clean build files:

    `make clean`

* Clean output folder:

    `make clean-output`

## Run

Create `output` folder, then run:

* If built with CUDA codec:

    `./graphics-cuda.out`

* If built with LodePNG codec:

    `./graphics-lode.out`
