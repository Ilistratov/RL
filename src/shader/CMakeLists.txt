set(SRC
  loader.cpp
)

add_library(shader OBJECT ${SRC})
target_link_libraries(shader PRIVATE rl_common)
