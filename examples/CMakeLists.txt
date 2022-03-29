cmake_minimum_required (VERSION 3.8)

project(Examples)

set(SRC
  mandelbrot.cpp
  raytracer.cpp
  swapchain_present_pass.cpp
)

add_library(examples ${SRC})
target_link_libraries(examples rl_common rl_lib)
