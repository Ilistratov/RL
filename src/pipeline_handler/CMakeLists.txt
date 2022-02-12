set(SRC
  compute.cpp
)

add_library(pipeline_handler ${SRC})
target_link_libraries(pipeline_handler rl_common)
