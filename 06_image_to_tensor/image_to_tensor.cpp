#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "     OPENCV TO LIBTORCH CUDA PIPELINE    " << std::endl;
    std::cout << "=========================================" << std::endl;

    // 1. Load a real image using OpenCV (BGR Format)
    std::string image_path = "/home/sameer/Documents/cpp_ai_roadmap/06_image_to_tensor/test.jpg"; // image path (update if needed)
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);

    if (img.empty()) {
        std::cerr << "[ERROR] Could not load image: " << image_path << std::endl;
        std::cerr << "[TIP] Please update the image path!" << std::endl;
        return -1;
    }
    std::cout << "[SUCCESS] Loaded image size: [" << img.rows << "x" << img.cols << "]" << std::endl;

    // STEP 1: The Memory Bridge (Wrap Tensor around OpenCV raw pointer)
    // Shape created: [Height, Width, Channels] -> Type: Unsigned 8-bit Integer
    torch::Tensor tensor_img = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);

    // STEP 2: The Shape Shifter (Permute dimensions for Deep Learning layouts)
    // Converts layout from: [Height (0), Width (1), Channels (2)] 
    // To deep learning format: [Channels (2), Height (0), Width (1)]
    tensor_img = tensor_img.permute({2, 0, 1});

    // STEP 3: The GPU Launchpad (Convert to float, normalize, and push to CUDA VRAM)
    // AI networks expect floats between 0.0 and 1.0 instead of 0 to 255 integers
    torch::Tensor cuda_tensor = tensor_img.to(torch::kFloat32).div(255.0).to(torch::kCUDA);

    // Verify the pipeline results via terminal diagnostics
    std::cout << "\n--- Final Tensor Diagnostics ---" << std::endl;
    std::cout << "Tensor Object Data Type : " << cuda_tensor.dtype() << std::endl;
    std::cout << "Tensor Dimensions Count : " << cuda_tensor.dim() << std::endl;
    std::cout << "Tensor Shape Layout     : " << cuda_tensor.sizes() << " -> [Channels, Height, Width]" << std::endl;
    std::cout << "Active Target Device ID : " << cuda_tensor.device() << std::endl;
    std::cout << "=========================================" << std::endl;

    return 0;
}