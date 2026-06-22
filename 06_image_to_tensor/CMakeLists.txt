cmake_minimum_required(VERSION 3.22)
project(ImageToTensor CXX)

# Set C++ standard to modern C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Locate LibTorch SDK
set(Torch_DIR "/home/sameer/Documents/cpp_ai_roadmap/libtorch_sdk/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# 2. Locate System OpenCV
find_package(OpenCV REQUIRED)

# Create the executable target binary
add_executable(image_to_tensor_app image_to_tensor.cpp)

# Link both LibTorch and OpenCV libraries to the application binary
target_link_libraries(image_to_tensor_app "${TORCH_LIBRARIES}" ${OpenCV_LIBS})

# Apply the NVSHMEM cluster bypass fix to this target
target_link_options(image_to_tensor_app PRIVATE "-Wl,--allow-shlib-undefined")