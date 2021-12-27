cmake_minimum_required (VERSION 3.8)

project(RL)
include(BuildFlags.cmake)
message(STATUS "Using compiler: ${CMAKE_CXX_COMPILER_ID}")

string(LENGTH "${CMAKE_SOURCE_DIR}/" SOURCE_PATH_SIZE)
add_definitions("-DSOURCE_PATH_SIZE=${SOURCE_PATH_SIZE}")

add_library(RLCommon INTERFACE)

target_include_directories(RLCommon INTERFACE ${CMAKE_SOURCE_DIR})
target_include_directories(RLCommon SYSTEM INTERFACE ${PROJECT_SOURCE_DIR}/third_party/include)
target_include_directories(RLCommon SYSTEM INTERFACE $ENV{VULKAN_SDK}/Include)

target_link_directories(RLCommon INTERFACE ${PROJECT_SOURCE_DIR}/third_party/lib)
target_link_directories(RLCommon INTERFACE $ENV{VULKAN_SDK}/Lib)

target_link_libraries(RLCommon INTERFACE vulkan-1)
target_link_libraries(RLCommon INTERFACE glfw3)
add_compile_definitions(VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)

target_compile_options(RLCommon INTERFACE -Wall)

set(RL_SRC
	main.cpp
)

add_executable(RL ${RL_SRC})
target_link_libraries(${PROJECT_NAME} RLCommon)

add_subdirectory(base)
target_link_libraries(RL Base)

add_subdirectory(utill)
target_link_libraries(RL Utill)
