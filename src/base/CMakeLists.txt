set(SRC
  base.cpp
  context.cpp
  physical_device_picker.cpp
  swapchain.cpp
  window.cpp
)

add_library(base OBJECT ${SRC})
target_link_libraries(base PRIVATE rl_common)
