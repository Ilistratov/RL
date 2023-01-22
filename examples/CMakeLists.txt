cmake_minimum_required (VERSION 3.8)

project(Examples)

set(SRC
  blit_to_swapchain.cpp
  common.cpp
  mandelbrot.cpp
  raytracer.cpp
  raytracer-2.cpp
  test.cpp
)

add_library(examples ${SRC})
target_link_libraries(examples rl_common rl_lib)
