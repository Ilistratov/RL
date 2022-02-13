set(SRC
  logger.cpp
)

add_library(utill ${SRC})
target_link_libraries(utill rl_common)
