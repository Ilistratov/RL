set(SRC
  command_pool.cpp
  executor.cpp
  timeline_semaphore.cpp
)

add_library(gpu_executor OBJECT ${SRC})
target_link_libraries(gpu_executor PRIVATE rl_common)
