set(GpuExecuter_SRC
  executer.cpp
  task.cpp
  timeline_semaphore.cpp
)

add_library(GpuExecuter ${GpuExecuter_SRC})
target_link_libraries(GpuExecuter RLCommon)
