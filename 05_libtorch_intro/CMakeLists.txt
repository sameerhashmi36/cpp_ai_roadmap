cmake_minimum_required(VERSION 3.22)
project(LibTorchIntro CXX)

# Set C++ standard to modern C++17
set(CMAKE_CXX_STANDARD 17)

# Tell CMake exactly where our downloaded LibTorch folder is hidden
set(Torch_DIR "/home/sameer/Documents/cpp_ai_roadmap/libtorch_sdk/libtorch/share/cmake/Torch")

# Look for the Torch package automatically using the directory path above
find_package(Torch REQUIRED)

# Create our executable target binary
add_executable(tensor_app tensor_basics.cpp)

# Link the massive LibTorch libraries to our application binary
target_link_libraries(tensor_app "${TORCH_LIBRARIES}")