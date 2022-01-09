set(GpuResources_SRC
  buffer_manager.cpp
  buffer.cpp
  device_memory_allocator.cpp
  image_manager.cpp
  image.cpp
  memory_block.cpp
)

add_library(GpuResources ${GpuResources_SRC})
target_link_libraries(GpuResources RLCommon)
