/*
 * PROJECT: Phase 5 - LibTorch Device Management & Hardware Acceleration
 * DESCRIPTION: Queries system infrastructure for CUDA compatibility, dynamically 
 * routes tensor allocation blocks, and runs accelerated matrix multiplication.
 * INPUT: Hardcoded evaluation dimensions.
 * OUTPUT: Hardware diagnostic flags and matrix processing execution telemetry.
*/

#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "    LIBTORCH DEVICE & ACCELERATION STUDY " << std::endl;
    std::cout << "=========================================" << std::endl;

    // 1. HARDWARE DETECTION
    // Python equivalent: is_cuda = torch.cuda.is_available()
    bool cuda_available = torch::cuda::is_available();
    std::cout << "NVIDIA CUDA Driver Detection Status: " 
              << (cuda_available ? "FOUND (GPU Accelerated)" : "NOT FOUND (Fallback to CPU)") 
              << std::endl;

    // 2. EXPLICIT ROUTING VARIABLE
    // Setup a dynamic device mailbox mapping target based on system capability
    torch::Device target_device(torch::kCPU); // Default fallback target

    if (cuda_available) {
        target_device = torch::Device(torch::kCUDA);
        std::cout << "Active Execution Target Bus: NVIDIA VRAM Vanks" << std::endl;
    } else {
        std::cout << "Active Execution Target Bus: System RAM Buffer" << std::endl;
    }

    // 3. TENSOR ALLOCATION ON THE SELECT DEVICE
    // create a 1000x1000 weight matrix directly on target hardware memory block
    std::cout << "\nAllocating dense matrices directly onto device target..." << std::endl;
    
    // pass the device target configuration directly as an option parameter
    torch::Tensor matrix_A = torch::randn({1000, 1000}, torch::TensorOptions().device(target_device));
    torch::Tensor matrix_B = torch::randn({1000, 1000}, torch::TensorOptions().device(target_device));

    // 4. ACCELERATED MATRIX MULTIPLICATION
    // Python equivalent: matrix_C = torch.mm(matrix_A, matrix_B)
    std::cout << "Executing intensive parallelized matrix multiplication (A x B)..." << std::endl;
    
    // Executing via optimized hardware kernels (BLAS on CPU, cuBLAS on GPU)
    torch::Tensor matrix_C = torch::mm(matrix_A, matrix_B);

    // 5. INSPECTION SUMMARY
    std::cout << "\n--- Calculation Complete ---" << std::endl;
    std::cout << "Result Matrix Shape Summary: " << matrix_C.sizes() << std::endl;
    std::cout << "Result Matrix Device Anchor: " << matrix_C.device() << std::endl;

    return 0;
}