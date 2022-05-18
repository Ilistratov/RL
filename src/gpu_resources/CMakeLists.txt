set(SRC
  access_sync_manager.cpp
  buffer.cpp
  device_memory_allocator.cpp
  image.cpp
  memory_block.cpp
  resource_manager.cpp
)

add_library(gpu_resources OBJECT ${SRC})
target_link_libraries(gpu_resources PRIVATE rl_common)
