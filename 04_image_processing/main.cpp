#include "ImageEngine.h"
#include <iostream>

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "   STARTING AI VISION PREPROCESSING APP   " << std::endl;
    std::cout << "=========================================" << std::endl;

    // 1. Instantiate the Object
    // This allocates memory for the class and runs the constructor.
    // Equivalent to Python's: engine = ImageEngine()
    ImageEngine engine;

    // 2. Capture Input Data
    // This will either grab the real webcam frame or generate the synthetic matrix data
    engine.captureInput();

    // 3. Apply Data Augmentation
    // We pass a positive value to increase pixel intensities (simulating direct bright light adjustment)
    engine.applyBrightness(100.0);

    // 4. Export the final matrix payload out to a physical file
    std::string output_path = "processed_output.jpg";
    engine.saveResult(output_path);

    std::cout << "\n=========================================" << std::endl;
    std::cout << "   PIPELINE COMPLETE: Check " << output_path << std::endl;
    std::cout << "=========================================" << std::endl;

    return 0;
}