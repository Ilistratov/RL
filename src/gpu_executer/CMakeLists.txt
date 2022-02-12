set(GpuExecuter_SRC
  command_pool.cpp
  executer.cpp
  timeline_semaphore.cpp
)

add_library(GpuExecuter ${GpuExecuter_SRC})
target_link_libraries(GpuExecuter RLCommon)
