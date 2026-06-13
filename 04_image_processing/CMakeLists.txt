cmake_minimum_required(VERSION 3.22)
project(ImageProcessingEngine CXX)

# Set C++ standard to modern C++17
set(CMAKE_CXX_STANDARD 17)

# Look for OpenCV on your Ubuntu system automatically
find_package(OpenCV REQUIRED)

# Include the OpenCV directories so the compiler can see them
include_directories(${OpenCV_INCLUDE_DIRS})

# Tell CMake about our multi-file layout
add_executable(vision_app 
    main.cpp 
    ImageEngine.cpp 
    ImageEngine.h
)

# Link the core OpenCV engine binaries to our app
target_link_libraries(vision_app ${OpenCV_LIBS})