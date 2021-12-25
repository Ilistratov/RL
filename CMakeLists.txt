cmake_minimum_required (VERSION 3.8)

project(RL)
include(BuildFlags.cmake)
message(STATUS "Using compiler: ${CMAKE_CXX_COMPILER_ID}")

add_library(RendererCommonDependencies INTERFACE)

target_include_directories(RendererCommonDependencies INTERFACE ${CMAKE_SOURCE_DIR})
target_include_directories(RendererCommonDependencies SYSTEM INTERFACE ${PROJECT_SOURCE_DIR}/third_party/include)
target_include_directories(RendererCommonDependencies SYSTEM INTERFACE $ENV{VULKAN_SDK}/Include)

target_link_directories(RendererCommonDependencies INTERFACE ${PROJECT_SOURCE_DIR}/third_party/lib)
target_link_directories(RendererCommonDependencies INTERFACE $ENV{VULKAN_SDK}/Lib)

target_link_libraries(RendererCommonDependencies INTERFACE vulkan-1)
target_link_libraries(RendererCommonDependencies INTERFACE glfw3)
add_compile_definitions(VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)

target_compile_options(RendererCommonDependencies INTERFACE -Wall)

set(SRC
	main.cpp
)

add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} RendererCommonDependencies)
