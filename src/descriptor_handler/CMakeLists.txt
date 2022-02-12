set(SRC
  binding.cpp
  pool.cpp
  set.cpp
)

add_library(descriptor_handler ${SRC})
target_link_libraries(descriptor_handler rl_common)
