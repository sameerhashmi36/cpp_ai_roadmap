# 05_LibTorch_Intro: PyTorch C++ Backend

This module introduces **High-Performance C++ Inference**. By using LibTorch, it interact directly with the C++ backend that powers PyTorch.

## Key Achievements
* **Library Integration**: Successfully linked the `LibTorch` SDK using CMake.
* **Tensor Allocation**: Created dynamic tensors using `torch::randn` and `torch::ones`.
* **Metadata Inspection**: Programmatically accessed tensor shapes (`sizes()`), element counts (`numel()`), and data types (`dtype()`).
* **Memory Views**: Performed non-destructive reshaping of data using `.view()`.

## Build Instructions
Here is exactly how I initialized, linked, and executed my first LibTorch application:

1. **SDK Installation & Environment Setup**:
To isolate the neural network infrastructure dependencies, that pulled down the pre-compiled LibTorch binaries from the official PyTorch distribution channels:
```bash
mkdir -p libtorch_sdk && cd libtorch_sdk

wget [https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip](https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip)

unzip libtorch-shared-with-deps-latest.zip && rm libtorch-shared-with-deps-latest.zip
```

2. **Configure**: Ensure the `Torch_DIR` in `CMakeLists.txt` points to the `libtorch_sdk` folder.

3. **Compile**:
```bash
   mkdir -p build && cd build
   cmake ..
   make

   ./tensor_app
```