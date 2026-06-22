# Module 07: Deep Learning Model Inference (Classification & Object Detection)

This module serves as the production-grade bridge between Python model prototyping and high-performance C++ deployment. It demonstrates how to compile both standard image classification networks (ResNet18) and real-time object detectors (YOLOv8) into standalone TorchScript computational graphs, ingest real data streams using OpenCV, and execute hardware-accelerated forward passes natively on an NVIDIA GPU.

---

## Core Architecture

This workspace contains two separate production pipelines sharing a common LibTorch and OpenCV build infrastructure:

### 1. Image Classification Pipeline (`model_inference_app`)
* **Model:** ResNet18 (Pre-trained on ImageNet).
* **Graph Compilation:** Traced in Python with a standard CPU tensor footprint.
* **Inference Pipeline:** Resizes input images to $224 \times 224$, structures them into a 4D tensor `[1, 3, 224, 224]`, evaluates raw category confidence logits on the GPU, and outputs the top predicted mathematical category index via an `.argmax()` extraction.

### 2. Object Detection Pipeline (`object_detection_app`)
* **Model:** Ultralytics YOLOv8 Nano (Pre-trained on 80 COCO classes).
* **Graph Compilation:** Traced directly on `device=0` (GPU) to bake internal anchor grid matrices into VRAM coordinates, bypassing TorchScript runtime device mismatches (`RuntimeError: Expected all tensors to be on the same device`).
* **Inference & Post-Processing Engine:** Resizes images to $640 \times 640$, processes a complex matrix slice of $8400$ candidate object regions on the GPU, uses OpenCV's built-in **Non-Maximum Suppression (NMS)** algorithm to filter overlapping duplicate boundaries, and overlays labeled neon-green bounding boxes directly onto the original source image matrix.

---

## Technical Specifications

* **C++ Standard:** C++17
* **Primary Dependencies:** LibTorch (CUDA 12.6 backend compiled via GCC-12 compatibility layers), OpenCV 4.x
* **Execution Target Hardware:** NVIDIA GPU VRAM (`cuda:0`)

---

## Workspace Directory Map

```text
07_model_inference/
├── CMakeLists.txt            # Unified CMake file managing both target binaries
├── export_model.py           # ResNet18 compilation script (Python)
├── export_yolo.py            # YOLOv8 GPU-anchored compilation script (Python)
├── model_inference.cpp       # Image classification C++ driver
├── object_detection.cpp      # Object detection and overlay C++ driver
└── build/                    # Native build tree target location
    ├── resnet18.pt           # Compiled ResNet graph asset
    └── yolov8n.torchscript   # Compiled YOLO graph asset
```

---

## Model Compilation Phase (Python Environment)

```
# on the path of 07_model_inference
python export_model.py
python export_yolo.py
```

---

## Native Compilation Phase (C++ Environment)

```
cd build/

cmake -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 ..

make -j$(nproc)
```

---

## Execution Phase

```
# For classification logits:
./model_inference_app

# For Object detection:
./object_detection_app
```