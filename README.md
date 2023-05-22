# RL

## Build

### Windows

You will need:

- VulkanSDK 1.3 or later with **GLM** and **VMA** headers, can be found [here](https://vulkan.lunarg.com/sdk/home)
- CMake 3.20 or later
- GLFW (for this project version 3.3.8 was used)
- tiny_obj_loader.h header
- ninja or other build tool of your choice

GLFW and tiny_obj_loader can be downloaded from [here](https://disk.yandex.com/d/B8wdbdlKlPapjw). Contents of the archive should be extracted in to the repo root. (archive also contains one of the scenes used for perf-tests)

Once everything is ready, create build folder and run
```
cmake ..
ninja install
```
(tested with clang 15.0.6)

### Run

Simply run renderer executable while in the install/bin directory.
