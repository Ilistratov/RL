set(SRC
  command_pool.cpp
  executer.cpp
  timeline_semaphore.cpp
)

add_library(gpu_executer OBJECT ${SRC})
target_link_libraries(gpu_executer PRIVATE rl_common)
