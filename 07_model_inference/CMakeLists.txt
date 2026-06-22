cmake_minimum_required(VERSION 3.22)
project(ModelInference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Locate Dependencies
set(Torch_DIR "/home/sameer/Documents/cpp_ai_roadmap/libtorch_sdk/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# TARGET 1: Image Classification (ResNet18)
add_executable(model_inference_app model_inference.cpp)
target_link_libraries(model_inference_app "${TORCH_LIBRARIES}" ${OpenCV_LIBS})
target_link_options(model_inference_app PRIVATE "-Wl,--allow-shlib-undefined")

# TARGET 2: Object Detection (YOLOv8) - NEW!
add_executable(object_detection_app object_detection.cpp)
target_link_libraries(object_detection_app "${TORCH_LIBRARIES}" ${OpenCV_LIBS})
target_link_options(object_detection_app PRIVATE "-Wl,--allow-shlib-undefined")