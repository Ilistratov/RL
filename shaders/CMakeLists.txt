cmake_minimum_required (VERSION 3.8)

set(HLSL_SOURCE_FILES
	${CMAKE_CURRENT_SOURCE_DIR}/mandelbrot.hlsl
)

foreach(HLSL ${HLSL_SOURCE_FILES})
	get_filename_component(FILE_NAME ${HLSL} NAME_WE)
	set(SPIRV ${PROJECT_BINARY_DIR}/${FILE_NAME}.spv)
	add_custom_command(
		OUTPUT ${SPIRV}
		COMMAND $ENV{VK_SDK_PATH}/bin/dxc.exe -spirv -T cs_6_7  -fspv-target-env=vulkan1.1 -E main -Fo ${SPIRV} ${HLSL} &&
				$ENV{VK_SDK_PATH}/bin/spirv-val.exe ${SPIRV}
		DEPENDS ${HLSL}
	)
	list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(HLSL)

add_custom_target(shaders DEPENDS ${SPIRV_BINARY_FILES})
