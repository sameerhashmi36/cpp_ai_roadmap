# Module 06: OpenCV Image to LibTorch CUDA Tensor Pipeline

This project implements an optimized production-grade computer vision data pipeline in C++. It demonstrates how to ingest a real-world image from disk using OpenCV, map its raw memory block into a LibTorch tensor without redundant memory allocation, restructure its spatial dimensions, and offload the data onto an NVIDIA GPU via CUDA.

---

## Pipeline Architecture

Deep learning models require data to be formatted in a highly specific configuration. This application orchestrates a 3-step translation process between the CPU image processor and the GPU AI engine:

1. **The Memory Bridge (`torch::from_blob`):** Automatically wraps a LibTorch tensor directly around OpenCV's raw data pointer (`img.data`). This prevents copying heavy pixel matrices across standard system memory.
2. **The Shape Shifter (`.permute`):** Switches the standard OpenCV layout of `[Height, Width, Channels]` to the industry-standard deep learning channel-first layout of `[Channels, Height, Width]`.
3. **The GPU Launchpad (`.to`):** Casts the unsigned 8-bit integers (`0` to `255`) into floating-point numbers (`0.0` to `1.0`), normalizes them by dividing by `255.0`, and ships the final matrix structure directly into GPU VRAM (`cuda:0`).

> **Note on Batching:** Real inference engines expect a 4D tensor format containing `[Batch, Channels, Height, Width]`. To prep this tensor for an actual neural network, apply `.unsqueeze(0)` to append the explicit batch dimension.

---

## Technical Specifications

* **C++ Standard:** C++17
* **Core Dependecies:** LibTorch (with CUDA 12.6 support), OpenCV (System-wide)
* **Target Execution Unit:** NVIDIA VRAM (`cuda:0`)

---

## Build and Execution Guide

### 1. Place a Test Image
Ensure a valid image file named `test.jpg` is present inside the folder, or use the automated retrieval command:
```bash
cd ~/Documents/cpp_ai_roadmap/06_image_to_tensor/build
cmake -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 ..
make -j$(nproc)
./image_to_tensor_app
```

