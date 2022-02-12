set(SRC
  base.cpp
  context.cpp
  physical_device_picker.cpp
  swapchain.cpp
  window.cpp
)

add_library(base ${SRC})
target_link_libraries(base rl_common)
