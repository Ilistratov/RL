cmake_minimum_required (VERSION 3.8)

project(RL_Lib)

add_library(rl_common INTERFACE)

target_include_directories(rl_common INTERFACE ${PROJECT_SOURCE_DIR})
target_include_directories(rl_common SYSTEM INTERFACE ${PROJECT_SOURCE_DIR}/third_party/include)
target_include_directories(rl_common SYSTEM INTERFACE $ENV{VULKAN_SDK}/Include)

target_link_directories(rl_common INTERFACE ${PROJECT_SOURCE_DIR}/third_party/lib)
target_link_directories(rl_common INTERFACE $ENV{VULKAN_SDK}/Lib)

target_link_libraries(rl_common INTERFACE vulkan-1)
target_link_libraries(rl_common INTERFACE glfw3)

target_compile_options(rl_common INTERFACE -Wall)

set(SUB_LIBS
  base
  gpu_executer
  gpu_resources
  pipeline_handler
  render_data
  render_graph
  utill
)

foreach(LIB ${SUB_LIBS})
  add_subdirectory(${LIB})
  list(APPEND OBJ_LIST $<TARGET_OBJECTS:${LIB}>)
endforeach()

add_library(rl_lib ${OBJ_LIST})
