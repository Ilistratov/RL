set(SRC
  mesh.cpp
)

add_library(render_data ${SRC})
target_link_libraries(render_data rl_common gpu_resources utill)
