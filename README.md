# High-Performance C++ for AI & Computer Vision

This repository tracks my step-by-step progression from standard Python/PyTorch prototyping into high-performance C++ production deployment, memory management, and GPU-accelerated inference. 

Rather than relying on interpreted scripts, this roadmap focuses on compiling standalone binaries that interface directly with hardware drivers, system memory buffers, and NVIDIA CUDA cores to deploy production-grade vision models.

---

## Repository Map (Current Progress)

| Module | Focus | Key Concepts & Implementations |
| :--- | :--- | :--- |
| **`01_basics`** | C++ Fundamentals for ML | Static typing (`float` vs `double`), compiled entry points, semicolons, and streams (`std::cout`). |
| **`02_memory`** | Hardware Memory Mechanics | Pass-by-value, reference, and pointers. Structuring memory layouts to eliminate copy overhead for heavy tensors. |
| **`03_oop`** | Object-Oriented AI Components | Encapsulating neural network layers, protecting internal weights/biases (`private`), and using constructor initializer lists. |
| **`04_image_processing`** | OpenCV Ingestion & Build Systems | Compilation mechanics, single-file compilation (`g++` + `pkg-config`) vs. multi-file production builds via **CMake**. |
| **`05_libtorch_intro`** | PyTorch C++ Backend | Linking the LibTorch native SDK, tensor allocations, metadata inspection (`sizes()`, `numel()`), and memory views. |
| **`06_image_to_tensor`** | Zero-Copy Data Pipelines | Bridging OpenCV matrices to LibTorch tensors via `torch::from_blob()`, memory shape-shifting, and shifting to CUDA VRAM. |
| **`07_model_inference`** | Full-Pipeline GPU Inference | Loading TorchScript graphs into C++ engines. Side-by-side deployment of ResNet18 classification and YOLOv8 object detection with Non-Maximum Suppression (NMS). |
| **`08_video_stream_inference`** | Real-Time Video & Live Camera | Upgrading static inference to stream continuously from `.mp4` video files or live hardware webcams via command line at 30+ FPS. |

---

## Core Technical Core Competencies Developed

* **The Python-to-C++ Paradigm Shift:** Moving away from line-by-line interpretation to ahead-of-time (AOT) binary compilation using CMake.
* **Zero-Copy Memory Mapping:** Passing massive image buffers from camera frames straight into neural network tensors via shared raw pointers, bypassing expensive system RAM replication steps.
* **Device Coordination (CUDA):** Managing asynchronous data movement across the host (CPU) and device (GPU) boundaries, avoiding common TorchScript runtime device mismatches.
* **Production Pre & Post-Processing:** Implementing custom OpenCV resizing, normalization, matrix slicing, and standalone Non-Maximum Suppression (NMS) bounding-box rendering completely outside of Python ecosystems.

---

## Tech Stack & Hardware Specs

* **Language Standard:** C++17
* **Build Architecture:** CMake (Minimum 3.22)
* **Deep Learning Framework:** LibTorch SDK (PyTorch C++ Backend with CUDA 12.6 tracking)
* **Computer Vision Library:** OpenCV 4.x
* **Primary Host Toolchain:** GCC 12 Compatibility Layers

---

## Work in Progress
This roadmap is an active, living learning document. As I explore more advanced optimization strategies, hardware deployment paradigms, and production inference architectures, new implementation modules will be systematically integrated here.