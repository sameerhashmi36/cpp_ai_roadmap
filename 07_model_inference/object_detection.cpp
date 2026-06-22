#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

const std::vector<std::string> COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

int main() {
    std::cout << "YOLOv8 CUDA OBJECT DETECTION ENGINE" << std::endl;

    // 1. Load the exported TorchScript YOLOv8 Model
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("yolov8m.torchscript");
        module.to(torch::kCUDA);
        module.eval();
    } catch (const c10::Error& e) {
        std::cerr << "[ERROR] Could not load model file!" << std::endl;
        return -1;
    }
    std::cout << "[SUCCESS] YOLOv8 Engine configured on CUDA." << std::endl;

    // 2. Load the input image from the local build folder
    std::string image_path = "/home/sameer/Documents/cpp_ai_roadmap/07_model_inference/test.jpg";
    cv::Mat original_img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (original_img.empty()) {
        std::cerr << "[ERROR] Image not found at path: " << image_path << std::endl;
        return -1;
    }

    // 3. Preprocess to YOLO specifications (640x640)
    cv::Mat resized_img;
    cv::resize(original_img, resized_img, cv::Size(640, 640));
    
    torch::Tensor tensor_img = torch::from_blob(resized_img.data, {640, 640, 3}, torch::kByte).permute({2, 0, 1});
    torch::Tensor cuda_tensor = tensor_img.to(torch::kFloat32).div(255.0).to(torch::kCUDA).unsqueeze(0);

    // 4. Run Inference Pass
    torch::NoGradGuard no_grad;
    auto output = module.forward({cuda_tensor}).toTensor().cpu(); 
    
    output = output.squeeze(0); 
    output = output.transpose(0, 1); 

    float scale_x = (float)original_img.cols / 640.0f;
    float scale_y = (float)original_img.rows / 640.0f;

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    float score_threshold = 0.25f; 

    // 5. Extract bounding boxes from the tensor layout
    for (int i = 0; i < output.size(0); ++i) {
        auto row_data = output[i];
        auto scores = row_data.slice(0, 4, 84);
        auto max_score_tensor = torch::max(scores, 0);
        float max_score = std::get<0>(max_score_tensor).item<float>();

        if (max_score > score_threshold) {
            int class_id = std::get<1>(max_score_tensor).item<int>();

            float cx = row_data[0].item<float>() * scale_x;
            float cy = row_data[1].item<float>() * scale_y;
            float w  = row_data[2].item<float>() * scale_x;
            float h  = row_data[3].item<float>() * scale_y;

            int left = static_cast<int>(cx - (w / 2.0f));
            int top  = static_cast<int>(cy - (h / 2.0f));

            boxes.push_back(cv::Rect(left, top, static_cast<int>(w), static_cast<int>(h)));
            confidences.push_back(max_score);
            class_ids.push_back(class_id);
        }
    }

    // 6. Run Non-Maximum Suppression (NMS) to clear duplicates
    std::vector<int> indices;
    float nms_threshold = 0.45f;
    cv::dnn::NMSBoxes(boxes, confidences, score_threshold, nms_threshold, indices);

    std::cout << "[INFO] Detections Extracted: " << indices.size() << " valid items found." << std::endl;

    // 7. Overlay predictions on the final image canvas
    for (int idx : indices) {
        cv::Rect box = boxes[idx];
        float conf = confidences[idx];
        int class_id = class_ids[idx];

        cv::rectangle(original_img, box, cv::Scalar(0, 255, 0), 3);
        std::string label = COCO_CLASSES[class_id] + ": " + cv::format("%.2f", conf * 100) + "%";
        
        int baseLine;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
        cv::rectangle(original_img, cv::Point(box.x, box.y - label_size.height - 5),
                      cv::Point(box.x + label_size.width, box.y), cv::Scalar(0, 255, 0), cv::FILLED);

        cv::putText(original_img, label, cv::Point(box.x, box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    }

    // 8. Output file directly to disk
    cv::imwrite("/home/sameer/Documents/cpp_ai_roadmap/07_model_inference/output_detection.jpg", original_img);
    std::cout << "[SUCCESS] Image analysis complete! Saved result image to: build/output_detection.jpg" << std::endl;

    return 0;
}