cmake_minimum_required (VERSION 3.8)

project(RL)
message(STATUS "Using compiler: ${CMAKE_CXX_COMPILER_ID}")

add_library(RLCommon INTERFACE)

target_include_directories(RLCommon INTERFACE ${PROJECT_SOURCE_DIR})
target_include_directories(RLCommon SYSTEM INTERFACE ${PROJECT_SOURCE_DIR}/third_party/include)
target_include_directories(RLCommon SYSTEM INTERFACE $ENV{VULKAN_SDK}/Include)

target_link_directories(RLCommon INTERFACE ${PROJECT_SOURCE_DIR}/third_party/lib)
target_link_directories(RLCommon INTERFACE $ENV{VULKAN_SDK}/Lib)

target_link_libraries(RLCommon INTERFACE vulkan-1)
target_link_libraries(RLCommon INTERFACE glfw3)
add_compile_definitions(VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)

target_compile_options(RLCommon INTERFACE -Wall)

add_library(RL rl_main.cpp)
target_link_libraries(${PROJECT_NAME} RLCommon)

add_subdirectory(base)
target_link_libraries(RL Base)

add_subdirectory(descriptor_handler)
target_link_libraries(RL DescriptorHandler)

add_subdirectory(gpu_executer)
target_link_libraries(RL GpuExecuter)

add_subdirectory(gpu_resources)
target_link_libraries(RL GpuResources)

add_subdirectory(pipeline_handler)
target_link_libraries(RL PipelineHandler)

add_subdirectory(shaders)
add_dependencies(RL Shaders)

add_subdirectory(utill)
target_link_libraries(RL Utill)
