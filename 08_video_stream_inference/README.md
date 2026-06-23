# Module 08: Real-Time Video & Camera Inference Engine

This module implements a high-performance computer vision pipeline designed to ingest, process, and analyze continuous video data streams in real time. Moving beyond static images, this engine combines OpenCV's hardware frame decoding loops with a GPU-accelerated LibTorch backend running a pre-compiled YOLOv8 Nano model to execute live object tracking at 30+ FPS.

---

## Core Architecture & Features

This application establishes a continuous execution loop that manages data movement across the CPU-GPU boundary without inducing memory leaks or pipeline lag:

1. **Unified Stream Ingestion (`cv::VideoCapture`):** Leverages a single interface capable of processing both encoded file buffers (such as `.mp4` or `.avi`) and raw live video streams from local Linux hardware nodes (such as `/dev/video0`).
2. **Dynamic Source Parsing:** Implements native command-line parsing to switch runtime data sources automatically based on input arguments without requiring a recompilation.
3. **GPU Forward Frame Pipeline:** Sequentially resizes frames to $640 \times 640$, transforms byte streams into standard multi-channel floating-point tensors, pushes them into GPU VRAM (`cuda:0`), and computes forward inference passes.
4. **Live Execution Profiling:** Integrates high-resolution timers (`std::chrono`) directly into the processing loop to track total end-to-end latency per frame and overlay a real-time **Frames Per Second (FPS)** counter onto the visual output canvas.

---

## Technical Specifications

* **C++ Standard:** C++17
* **Primary Dependencies:** LibTorch (CUDA 12.6 backend), OpenCV 4.x
* **Deployment Hardware:** NVIDIA GPU VRAM (`cuda:0`)

---

## Workspace Directory Map

```text
08_video_stream_inference/
├── CMakeLists.txt            # Host build configurations
├── video_detection.cpp       # Unified stream inference engine C++ source
└── build/                    # Native build tree target location
    ├── sample.mp4            # Sample tracking video asset (downloaded)
    └── yolov8m.torchscript   # GPU-traced YOLOv8 binary model graph

```

---

## Environment Setup & asset synchronization

```
cd ~/Documents/cpp_ai_roadmap/08_video_stream_inference/build
cp ../../07_model_inference/build/yolov8m.torchscript .

# Native Compilation Phase
cmake -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 ..
make

# Execution Interface Modes
# Processing an offline media file stream
./video_detection_app sample.mp4

# Attaching to the primary system webcam
./video_detection_app 0

# Attaching to an external USB companion camera input
./video_detection_app 1

# Press the q key at any point to exit()
```
