set(SRC
  copy.cpp
  scan.cpp
  sort.cpp
)

add_library(mp_primitives OBJECT ${SRC})
target_link_libraries(mp_primitives PRIVATE rl_common)
