#include "ImageEngine.h"
#include <iostream>

// 1. THE CONSTRUCTOR IMPLEMENTATION
// Notice the 'ImageEngine::' before the constructor name. 
// This tells the compiler we are writing the logic for the blueprint in the header.
ImageEngine::ImageEngine() {
    std::cout << "[Engine Initialize] Image processing engine structures initialized." << std::endl;
}

// 2. INPUT CAPTURE IMPLEMENTATION
void ImageEngine::captureInput() {
    std::cout << "[Engine Action] Searching for video feed device..." << std::endl;
    
    cv::VideoCapture cap(0); // Attempt to open your webcam
    
    if (!cap.isOpened()) {
        std::cout << "[Warning] No physical webcam detected. Generating a synthetic image..." << std::endl;
        // Fallback: Create a synthetic 480x640 3-channel (BGR) solid gray matrix image (Value: 128)
        // We write directly into our private member variable 'raw_frame'
        raw_frame = cv::Mat(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    } else {
        std::cout << "[Success] Webcam connected. Capturing frame..." << std::endl;
        cap >> raw_frame; // Stream the webcam pixel matrix directly into 'raw_frame'
        cap.release();    // Release the camera hardware block back to the system
    }
}

// 3. BRIGHTNESS DATA AUGMENTATION IMPLEMENTATION
void ImageEngine::applyBrightness(double amount) {
    // Check if we have image data inside our private variable before processing
    if (raw_frame.empty()) {
        std::cerr << "[Error] Cannot modify brightness. No frame captured yet!" << std::endl;
        return;
    }

    std::cout << "[Engine Action] Altering frame pixel values by shifting brightness: " << amount << std::endl;
    
    // Matrix conversion: updates intensity values directly.
    // -1 keeps the exact same output pixel matrix data type (CV_8U bytes)
    // 1.0 means no contrast scaling, 'amount' shifts the raw brightness value up or down
    raw_frame.convertTo(raw_frame, -1, 1.0, amount);
}

// 4. IMAGE EXPORT FILE WRITER IMPLEMENTATION
void ImageEngine::saveResult(const std::string& filename) {
    if (raw_frame.empty()) {
        std::cerr << "[Error] Cannot save. Matrix content is empty!" << std::endl;
        return;
    }

    std::cout << "[Engine Action] Exporting processed matrix payload to disk: " << filename << std::endl;
    
    // OpenCV built-in file encoder writer saves the image matrix out as a file
    cv::imwrite(filename, raw_frame);
}