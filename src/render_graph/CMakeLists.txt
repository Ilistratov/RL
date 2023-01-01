set(SRC
  pass.cpp
  render_graph.cpp
)

add_library(render_graph OBJECT ${SRC})
target_link_libraries(render_graph PRIVATE rl_common)
