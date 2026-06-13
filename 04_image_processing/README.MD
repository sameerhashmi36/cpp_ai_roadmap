# 04_Image_Processing: Vision Pipelines & Compilation Mechanics

This module explores how production computer vision pipelines interface with hardware devices, apply data augmentations, and handle compilation architecture in C++ using **OpenCV 4**.

---

## The Core Concept: C++ Compilation vs. Python Interpretation

Unlike Python, which translates text scripts line-by-line during runtime, **C++ is a compiled language**. The CPU cannot read C++ source code directly. The code must be translated entirely ahead of time into a standalone binary file containing raw machine code instructions (`0`s and `1`s).

### Why do we need the build tools?
* **Single-File Track:** The compiler (`g++`) needs to be told exactly where external libraries like OpenCV live on Ubuntu. We use `pkg-config` to automatically fetch those folder paths.
* **Multi-File Track:** When code is split into headers (`.h`) and implementations (`.cpp`), the files are completely blind to each other. **CMake** acts as the project architect, telling the compiler how to stitch all files and libraries together into a single cohesive application.

---

## Track 1: The Single-File Camera Stream (`capture_main.cpp`)

This approach squeezes the entire pipeline into one file, bypassing CMake entirely by passing compilation flags directly to the compiler. It connects to the default video device and creates a continuous live execution loop utilizing `cv::imshow`.

### Compilation Command
From the `04_image_processing/` directory, run:
```bash
g++ capture_main.cpp -o live_capture `pkg-config --cflags --libs opencv4`
```

1. g++: Invokes the GNU C++ compiler engine.

2. -o live_capture: Names the resulting standalone binary executable file.

3. `pkg-config ...`: Queries Ubuntu to find and link the OpenCV header paths and binary engine modules.


### Running the App
```
./live_capture
```
(Press q or Esc on the keyboard while focusing the window to close the stream safely).


## Track 2: The Multi-File Modular Engine (Production Layout)
In production AI frameworks, logic is separated into distinct modules to maintain scalability and code cleanliness:

1. ImageEngine.h: The Blueprint (Declares class properties and access permissions).

2. ImageEngine.cpp: The Muscles (Implements real-time capture, data manipulation, and file exports).

3. main.cpp: The Conductor (Instantiates the class object and coordinates the execution pipeline).

### Compilation Commands (Using CMake)
Because this setup contains multiple files tracking across external libraries, we use CMake to manage the generation details safely inside an isolated environment:

```
# 1. Enter the isolated build folder
mkdir -p build && cd build

# 2. Let CMake scan the parent directory lists and find OpenCV
cmake ..

# 3. Bake all source objects into a final binary file
make
```

### Running the App

Once make reaches [100\%] Built target vision_app, execute the multi-file application:
```
./vision_app
```

## File Structure
```
04_image_processing/
├── CMakeLists.txt        # The build configuration recipe manual
├── ImageEngine.h         # Class declarations & encapsulation rules
├── ImageEngine.cpp       # Video capture & brightness augmentation logic
├── main.cpp              # Application entry point orchestrator
├── capture_main.cpp      # Isolated, single-file live webcam stream script
└── README.md             # Core concept architectural documentation
```