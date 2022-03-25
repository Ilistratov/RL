cmake_minimum_required (VERSION 3.8)
project(RL_Executable)

include(BuildFlags.cmake)

add_compile_definitions(VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)
add_compile_definitions(VULKAN_HPP_NO_EXCEPTIONS)
string(LENGTH "${PROJECT_SOURCE_DIR}/" SOURCE_PATH_SIZE)
add_definitions("-DSOURCE_PATH_SIZE=${SOURCE_PATH_SIZE}")

set(SRC
	main.cpp
)

add_executable(renderer ${SRC})

add_subdirectory(examples)
target_link_libraries(renderer examples)

add_subdirectory(shaders)
add_dependencies(renderer shaders)

add_subdirectory(src)
target_link_libraries(renderer rl_lib)
