set(SRC
  buffer_pass_bind.cpp
  image_pass_bind.cpp
  pass.cpp
  resource_manager.cpp
)

add_library(render_graph ${SRC})
target_link_libraries(render_graph rl_common base gpu_resources utill)