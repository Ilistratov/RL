set(PipelineHandler_SRC
  compute.cpp
)

add_library(PipelineHandler ${PipelineHandler_SRC})
target_link_libraries(PipelineHandler RLCommon)
