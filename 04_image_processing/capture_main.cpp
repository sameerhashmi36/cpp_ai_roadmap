#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::cout << "=== Opening Live Camera Stream ===" << std::endl;
    std::cout << "Press the 'q' key on your keyboard to exit!" << std::endl;

    // 1. Open the default camera stream (Index 0)
    cv::VideoCapture cap(0);

    // Safety check: Did the camera actually open?
    if (!cap.isOpened()) {
        std::cerr << "[Error] Could not open webcam device!" << std::endl;
        return -1;
    }

    // 2. Create a persistent window to display the video frames
    cv::namedWindow("Live Video Feed", cv::WINDOW_AUTOSIZE);

    // An empty Matrix container to hold each frame as it arrives
    cv::Mat frame;

    // 3. The Continuous Stream Loop (Equivalent to Python's: while True)
    while (true) {
        // Grab the latest frame from the camera stream and load it into our matrix
        cap >> frame;

        if (frame.empty()) {
            std::cout << "[Warning] Blank frame captured. Skipping..." << std::endl;
            break;
        }

        // Display the matrix frame inside our window (Just like cv2.imshow in Python)
        cv::imshow("Live Video Feed", frame);

        // Wait 30 milliseconds for a keypress. If 'q' is pressed, break the loop.
        // This waitKey function is mandatory; without it, the window will freeze!
        char key = (char)cv::waitKey(30);
        if (key == 'q' || key == 27) { // 27 is the Escape key code
            std::cout << "Closing camera stream..." << std::endl;
            break;
        }
    }

    // 4. Clean up and free memory resources
    cap.release();
    cv::destroyAllWindows();
    return 0;
}