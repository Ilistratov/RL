set(SRC
  bvh.cpp
  mesh.cpp
# transfer_scheduler.cpp
)

add_library(render_data OBJECT ${SRC})
target_link_libraries(render_data PRIVATE rl_common)
