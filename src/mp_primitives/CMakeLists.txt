set(SRC
  scan.cpp
)

add_library(mp_primitives OBJECT ${SRC})
target_link_libraries(mp_primitives PRIVATE rl_common)
