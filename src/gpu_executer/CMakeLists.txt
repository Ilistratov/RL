set(SRC
  command_pool.cpp
  executer.cpp
  timeline_semaphore.cpp
)

add_library(gpu_executer ${SRC})
target_link_libraries(gpu_executer rl_common base utill)
