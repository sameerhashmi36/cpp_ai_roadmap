# 03_OOP: Object-Oriented Programming for AI Components

This module maps how custom deep learning architectures (like individual layers, activation modules, or full neural networks) are structured using strict C++ class architectures.

---

## Architectural Mechanics

### 1. Encapsulation (`private` vs `public`)
* **`private`**: Used to safely lock down internal layer parameters such as raw data weights, biases, and structural channel dimension shapes. This guarantees no external training script can corrupt internal arrays accidentally.
* **`public`**: Exposes the execution interface, predominantly the layer's `forward()` execution pipeline and safe tracking methods.

### 2. Constructor Initializer Lists
C++ optimizes memory allocation by utilizing an explicit initialization sequence:
```text
DenseLayer(int in_dim) : input_features(in_dim) { ... }
                         └────────────────────┘
                       Initialized directly in RAM 
                       before the function block opens.