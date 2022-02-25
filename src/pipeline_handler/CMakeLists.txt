set(SRC
  compute.cpp
  descriptor_binding.cpp
  descriptor_pool.cpp
  descriptor_set.cpp
)

add_library(pipeline_handler ${SRC})
target_link_libraries(pipeline_handler rl_common base utill)
