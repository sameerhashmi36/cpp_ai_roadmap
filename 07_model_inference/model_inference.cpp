#include <torch/script.h> // Crucial header for loading TorchScript models
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>

int main() {
    std::cout << "LIBTORCH CUDA JIT INFERENCE ENGINE" << std::endl;
    
    // 1. Load the model binary what exported from Python
    std::string model_path = "/home/sameer/Documents/cpp_ai_roadmap/07_model_inference/build/resnet18.pt";
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path);
        // Move the entire model's neural layer weights straight into GPU VRAM
        module.to(torch::kCUDA);
    }
    catch (const c10::Error& e) {
        std::cerr << "[ERROR] Failed to load the model: " << model_path << std::endl;
        return -1;
    }
    std::cout << "[SUCCESS] TorchScript Model loaded and transferred to CUDA device." << std::endl;

    // 2. Load the input image using OpenCV
    // I am reusing the test image from module 06!
    std::string image_path = "/home/sameer/Documents/cpp_ai_roadmap/06_image_to_tensor/test.jpg";
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);

    if (img.empty()) {
        std::cerr << "[ERROR] Could not load image from path: " << image_path << std::endl;
        return -1;
    }

    // 3. Image Preprocessing: ResNet18 expects explicit 224x224 spatial dimensions
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(224, 224));

    // STEP 1 & 2: Wrap tensor around memory and permute layout to [Channels, Height, Width]
    torch::Tensor tensor_img = torch::from_blob(resized_img.data, {224, 224, 3}, torch::kByte);
    tensor_img = tensor_img.permute({2, 0, 1});

    // STEP 3: Normalize to [0.0, 1.0], cast to Float32, and push to GPU
    torch::Tensor cuda_tensor = tensor_img.to(torch::kFloat32).div(255.0).to(torch::kCUDA);

    // STEP 4: Add the missing Batch Dimension -> converts [3, 224, 224] to [1, 3, 224, 224]
    cuda_tensor = cuda_tensor.unsqueeze(0);

    // 4. Execute Live Model Inference
    // Wrap the tensor inside a vector of torch::jit::IValue inputs
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(cuda_tensor);

    std::cout << "[INFO] Executing forward pass on GPU..." << std::endl;
    
    // Disable gradient calculations for maximum execution speed during inference
    torch::NoGradGuard no_grad; 
    torch::Tensor output = module.forward(inputs).toTensor();

    // 5. Parse Inference Outputs
    // Run an argmax across the 1000 ImageNet classification logits to extract the top predicted class ID
    int predicted_class_id = output.argmax(1).item<int>();

    std::cout << "\n--- Inference Metrics Tracker ---" << std::endl;
    std::cout << "Output Tensor Dimensions : " << output.sizes() << " -> [Batch, Classes]" << std::endl;
    std::cout << "Target Device Vector Unit: " << output.device() << std::endl;
    std::cout << "Highest Score Class ID   : " << predicted_class_id << std::endl;

    return 0;
}