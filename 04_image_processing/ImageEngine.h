#ifndef IMAGE_ENGINE_H
#define IMAGE_ENGINE_H

#include <opencv2/opencv.hpp>
#include <string>

// We define a clean OOP Class for managing the vision operations
class ImageEngine {
private:
    // Encapsulation: The raw image frame is protected inside the class
    cv::Mat raw_frame;

public:
    // Constructor Blueprint
    ImageEngine();

    // Safe Interface Methods (APIs)
    void captureInput();
    void applyBrightness(double amount);
    void saveResult(const std::string& filename);
};

#endif // IMAGE_ENGINE_H