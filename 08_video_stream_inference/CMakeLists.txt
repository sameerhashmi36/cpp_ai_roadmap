cmake_minimum_required(VERSION 3.22)
project(VideoStreamInference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Locate dependencies
set(Torch_DIR "/home/sameer/Documents/cpp_ai_roadmap/libtorch_sdk/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Executable declaration
add_executable(video_detection_app video_detection.cpp)
target_link_libraries(video_detection_app "${TORCH_LIBRARIES}" ${OpenCV_LIBS})
target_link_options(video_detection_app PRIVATE "-Wl,--allow-shlib-undefined")