set(GpuResources_SRC
  device_memory_allocator.cpp
  memory_block.cpp
)

add_library(GpuResources ${GpuResources_SRC})
target_link_libraries(GpuResources RLCommon)
