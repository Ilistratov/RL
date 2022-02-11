set(GpuResources_SRC
  access_sync_manager.cpp
  device_memory_allocator.cpp
  logical_buffer.cpp
  logical_image.cpp
  memory_block.cpp
  physical_buffer.cpp
  physical_image.cpp
)

add_library(GpuResources ${GpuResources_SRC})
target_link_libraries(GpuResources RLCommon)
