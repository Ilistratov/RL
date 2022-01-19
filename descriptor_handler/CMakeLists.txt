set(DescriptorHandler_SRC
  binding.cpp
  pool.cpp
  set.cpp
)

add_library(DescriptorHandler ${DescriptorHandler_SRC})
target_link_libraries(DescriptorHandler RLCommon)
