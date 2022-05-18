set(SRC
  error_handling.cpp
  input_manager.cpp
  logger.cpp
  transform.cpp
)

add_library(utill OBJECT ${SRC})
target_link_libraries(utill PRIVATE rl_common)
