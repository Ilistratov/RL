set(SRC
  buffer.cpp
  common.cpp
  device_memory_allocator.cpp
  image.cpp
  memory_block.cpp
  pass_access_syncronizer.cpp
  physical_buffer.cpp
  physical_image.cpp
  resource_access_syncronizer.cpp
  resource_manager.cpp
)

add_library(gpu_resources OBJECT ${SRC})
target_link_libraries(gpu_resources PRIVATE rl_common)
