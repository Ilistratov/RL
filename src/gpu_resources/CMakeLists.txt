set(SRC
  access_sync_manager.cpp
  buffer.cpp
  device_memory_allocator.cpp
  image.cpp
  logical_buffer.cpp
  logical_image.cpp
  memory_block.cpp
  physical_buffer.cpp
  physical_image.cpp
  resource_manager.cpp
)

add_library(gpu_resources ${SRC})
target_link_libraries(gpu_resources rl_common base)
