set(GpuResources_SRC
  access_sync_manager.cpp
  device_memory_allocator.cpp
  image_manager.cpp
  image.cpp
  logical_buffer.cpp
  memory_block.cpp
  physical_buffer.cpp
  resource_manager.cpp
)

add_library(GpuResources ${GpuResources_SRC})
target_link_libraries(GpuResources RLCommon)
