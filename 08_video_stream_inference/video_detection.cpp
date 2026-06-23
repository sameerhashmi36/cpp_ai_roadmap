#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>

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

int main(int argc, char** argv) {
    std::cout << "YOLOv8 REAL-TIME VIDEO INFERENCE ENGINE" << std::endl;

    // 1. Load the pre-compiled GPU YOLOv8 model
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("yolov8m.torchscript");
        module.to(torch::kCUDA);
        module.eval();
    } catch (const c10::Error& e) {
        std::cerr << "[ERROR] Could not load 'yolov8m.torchscript'." << std::endl;
        return -1;
    }
    std::cout << "[SUCCESS] YOLOv8 Engine loaded cleanly onto CUDA device." << std::endl;

    // 2. Initialize Video Stream Source dynamically from command-line arguments
    cv::VideoCapture cap;

    if (argc > 1) {
        std::string argument = argv[1];
        
        // If the argument is a simple integer digit (like "0" or "1"), parse it as a camera ID
        if (argument.find_first_not_of("0123456789") == std::string::npos) {
            int camera_id = std::stoi(argument);
            std::cout << "[INFO] Attempting to hook live webcam device ID: " << camera_id << std::endl;
            cap.open(camera_id);
        } else {
            // Otherwise, treat the input argument as a standard file pathway string
            std::cout << "[INFO] Attempting to ingest media file path: " << argument << std::endl;
            cap.open(argument);
        }
    } else {
        // DEFAULT FALLBACK: If you run the app without typing any arguments, it defaults to webcam 0
        std::cout << "[INFO] No source argument provided. Defaulting to live webcam device index 0..." << std::endl;
        cap.open(0);
    }

    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Failed to initialize video or camera capture source feed." << std::endl;
        return -1;
    }

    cv::Mat frame;
    cv::Mat resized_img;
    
    // Threshold Configurations
    float score_threshold = 0.25f;
    float nms_threshold = 0.45f;

    std::cout << "[INFO] Starting Live Processing Loop. Press 'q' on the GUI window to exit." << std::endl;

    // 3. High-Speed Processing Loop
    while (true) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Capture individual continuous sequence frames
        cap >> frame;
        if (frame.empty()) {
            std::cout << "[INFO] Video stream completed or empty frame detected. Exiting loop." << std::endl;
            break;
        }

        // Preprocess to standard 640x640 shape configurations
        cv::resize(frame, resized_img, cv::Size(640, 640));
        
        // Push frame matrix memory layout directly into GPU VRAM
        torch::Tensor tensor_img = torch::from_blob(resized_img.data, {640, 640, 3}, torch::kByte).permute({2, 0, 1});
        torch::Tensor cuda_tensor = tensor_img.to(torch::kFloat32).div(255.0).to(torch::kCUDA).unsqueeze(0);

        // Execute GPU Forward Pass
        torch::NoGradGuard no_grad;
        auto output = module.forward({cuda_tensor}).toTensor().cpu().squeeze(0).transpose(0, 1);

        // Coordinate scaling map ratios
        float scale_x = (float)frame.cols / 640.0f;
        float scale_y = (float)frame.rows / 640.0f;

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;

        // Extract Candidate Bounding Objects
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

        // Run Non-Maximum Suppression (NMS) to eliminate overlaps
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, score_threshold, nms_threshold, indices);

        // Render Labeled Boxes onto Frame Canvas
        for (int idx : indices) {
            cv::Rect box = boxes[idx];
            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);

            std::string label = COCO_CLASSES[class_ids[idx]] + ": " + cv::format("%.2f", confidences[idx] * 100) + "%";
            int baseLine;
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::rectangle(frame, cv::Point(box.x, box.y - label_size.height - 5),
                          cv::Point(box.x + label_size.width, box.y), cv::Scalar(0, 255, 0), cv::FILLED);
            cv::putText(frame, label, cv::Point(box.x, box.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }

        // Calculate and Render Live Execution Metric Tracker (FPS)
        auto end_time = std::chrono::high_resolution_clock::now();
        double fps = 1000.0 / std::chrono::duration<double, std::milli>(end_time - start_time).count();
        std::string fps_label = "Performance: " + cv::format("%.1f", fps) + " FPS";
        cv::putText(frame, fps_label, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

        // Display frame immediately to standard OS Desktop Window
        cv::imshow("YOLOv8 High-Performance Stream Diagnostics", frame);

        // Break execution window context loop immediately if user hits the 'q' key
        if (cv::waitKey(1) == 'q') {
            std::cout << "[INFO] Termination request logged by user." << std::endl;
            break;
        }
    }

    // Safely free hardware device resources
    cap.release();
    cv::destroyAllWindows();
    std::cout << "=========================================" << std::endl;
    return 0;
}