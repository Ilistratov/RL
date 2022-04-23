set(SRC
  error_handling.cpp
  input_manager.cpp
  logger.cpp
  transform.cpp
)

add_library(utill ${SRC})
target_link_libraries(utill rl_common)
