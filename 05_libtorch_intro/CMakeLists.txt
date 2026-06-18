cmake_minimum_required(VERSION 3.22)
project(LibTorchIntro CXX)

# Set C++ standard to modern C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Tell CMake exactly where the downloaded LibTorch folder is hidden (Fixed the terminal truncation here)
set(Torch_DIR "/home/sameer/Documents/cpp_ai_roadmap/libtorch_sdk/libtorch/share/cmake/Torch")

# Look for the Torch package automatically using the directory path above
find_package(Torch REQUIRED)

# ==========================================
# TARGET 1: Basic Tensor Operations Application
# ==========================================
add_executable(tensor_app tensor_basics.cpp)
target_link_libraries(tensor_app "${TORCH_LIBRARIES}")

# Force the linker to bypass NVSHMEM's cluster omissions for this target
target_link_options(tensor_app PRIVATE "-Wl,--allow-shlib-undefined")


# ==========================================
# TARGET 2: The hardware device acceleration application
# ==========================================
add_executable(device_math_app device_math.cpp)
target_link_libraries(device_math_app "${TORCH_LIBRARIES}")

# Force the linker to bypass NVSHMEM's cluster omissions for this target as well
target_link_options(device_math_app PRIVATE "-Wl,--allow-shlib-undefined")