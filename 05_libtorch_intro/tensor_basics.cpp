/*
 * PROJECT: Phase 5 - LibTorch Core Tensor Exploration
 * DESCRIPTION: Demonstrates tensor allocation, shape tracking, data type inspections, 
 * and mathematical matrix transformations using PyTorch's native C++ backend framework.
 * INPUT: Initialized float matrix buffers.
 * OUTPUT: Standard tensor structural summaries printed directly to terminal output streams.
 */

#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "   THE LIBTORCH TENSOR STUDY             " << std::endl;
    std::cout << "=========================================" << std::endl;

    // 1. ALLOCATION: Create a 2x3 matrix filled with random numbers
    // Python equivalent: x = torch.randn(2, 3)
    // In C++, matrix dimensions are passed inside curly braces {}
    torch::Tensor x = torch::randn({2, 3});
    std::cout << "Tensor X:\n" << x << std::endl;

    // 2. INSPECTION: Check structural details (Sizes and Data Types)
    // Python equivalents: x.size(), x.dtype
    std::cout << "\n--- Tensor Properties ---" << std::endl;
    std::cout << "Tensor Dimensions (Shapes): " << x.sizes() << std::endl;
    std::cout << "Number of Elements (Count): " << x.numel() << std::endl;
    std::cout << "Data Type Representation Code: " << x.dtype() << std::endl;

    // 3. TRANSFORMATIONS: Basic Matrix Math
    // Create a matrix filled completely with 1.0 values matching X's dimensions
    // Python equivalent: y = torch.ones(2, 3)
    torch::Tensor y = torch::ones({2, 3});
    
    // Element-wise matrix addition
    // Python equivalent: z = x + y
    torch::Tensor z = x + y;
    std::cout << "\nTensor Z (X + 1.0 Matrix):\n" << z << std::endl;

    // 4. ADVANCED OPERATIONS: Reshaping / Flattening
    // Python equivalent: flattened = z.view(-1) or z.view(6)
    torch::Tensor flattened = z.view({-1});
    std::cout << "\nFlattened 1D View of Tensor Z:\n" << flattened << std::endl;

    return 0;
}