cmake_minimum_required (VERSION 3.8)

project(RL_Lib)
message(STATUS "Using compiler: ${CMAKE_CXX_COMPILER_ID}")

add_library(rl_common INTERFACE)

target_include_directories(rl_common INTERFACE ${PROJECT_SOURCE_DIR})
target_include_directories(rl_common SYSTEM INTERFACE ${PROJECT_SOURCE_DIR}/third_party/include)
target_include_directories(rl_common SYSTEM INTERFACE $ENV{VULKAN_SDK}/Include)

target_link_directories(rl_common INTERFACE ${PROJECT_SOURCE_DIR}/third_party/lib)
target_link_directories(rl_common INTERFACE $ENV{VULKAN_SDK}/Lib)

target_link_libraries(rl_common INTERFACE vulkan-1)
target_link_libraries(rl_common INTERFACE glfw3)
add_compile_definitions(VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)

target_compile_options(rl_common INTERFACE -Wall)

add_library(rl_lib rl_main.cpp)
target_link_libraries(rl_lib rl_common)

add_subdirectory(base)
target_link_libraries(rl_lib base)

add_subdirectory(descriptor_handler)
target_link_libraries(rl_lib descriptor_handler)

add_subdirectory(gpu_executer)
target_link_libraries(rl_lib gpu_executer)

add_subdirectory(gpu_resources)
target_link_libraries(rl_lib gpu_resources)

add_subdirectory(pipeline_handler)
target_link_libraries(rl_lib pipeline_handler)

add_subdirectory(utill)
target_link_libraries(rl_lib utill)
