## Project Structure

* `01_basics/` - Fundamentals of static data types, memory footprints, and simple activation functions.

---

## Technical Insights for a Python Developer

Transitioning from an interpreted language to a compiled environment introduces key structural shifts:

### 1. Syntax & Core Mechanisms
* **`#include <iostream>`**: This acts similarly to a Python `import`. It brings in the standard input/output stream header file, granting access to console printing capabilities.
* **`int main()`**: Every executable C++ application must contain a root-level function named `main` that returns an integer. This serves as the absolute entry point for the operating system.
* **The Semicolon `;`**: Unlike Python's newline-based syntax, every single execution statement in C++ must conclude with a semicolon. The compiler uses this to explicitly parse where an instruction finishes.
* **`std::cout` and `std::endl`**: `cout` represents "character output" (the console stream), and `endl` represents "end line" (which inserts a newline character and flushes the stream buffer). The `<<` bitwise-shift operator is used to push sequential chunks of data into the output stream.

### 2. Core Data Types in ML Tensors
* **`double`**: A 64-bit floating-point type. Excellent for high-precision calculations like loss metrics or precise learning rates.
* **`float`**: A 32-bit floating-point type. This is the structural equivalent of `torch.float32` or standard NumPy floats. It is the dominant data type for weights, gradients, and activation calculations due to its optimized speed on hardware.