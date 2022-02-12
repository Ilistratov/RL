cmake_minimum_required (VERSION 3.8)
project(RL_Executable)

include(BuildFlags.cmake)

string(LENGTH "${PROJECT_SOURCE_DIR}/" SOURCE_PATH_SIZE)
add_definitions("-DSOURCE_PATH_SIZE=${SOURCE_PATH_SIZE}")

set(RL_SRC
	main.cpp
)

add_executable(RL_Executable ${RL_SRC})

add_subdirectory(src)
target_link_libraries(RL_Executable RL)
