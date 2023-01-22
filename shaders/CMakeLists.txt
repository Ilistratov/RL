cmake_minimum_required (VERSION 3.8)

set(HLSL_SOURCE_FILES
  mp-primitives/scan/aggregate.hlsl
  mp-primitives/scan/scatter.hlsl
	mandelbrot.hlsl
	raytrace.hlsl
	raytrace/debug-render.hlsl
	raytrace/raygen.hlsl
	raytrace/classify.hlsl
)

foreach(HLSL_SOURCE ${HLSL_SOURCE_FILES})
	set(hlsl_input ${CMAKE_CURRENT_SOURCE_DIR}/${HLSL_SOURCE})
	cmake_path(GET HLSL_SOURCE STEM spirv_output)
	cmake_path(GET HLSL_SOURCE PARENT_PATH output_dir)
	set(spirv_output ${PROJECT_BINARY_DIR}/shaders/${output_dir}/${spirv_output}.spv)
	add_custom_command(
		OUTPUT ${spirv_output}
		COMMAND $ENV{VK_SDK_PATH}/bin/dxc.exe -spirv -T cs_6_7 -HV 2021 -fspv-target-env=vulkan1.1 -E main -Fo ${spirv_output} ${hlsl_input} && $ENV{VK_SDK_PATH}/bin/spirv-val.exe ${spirv_output}
		DEPENDS ${hlsl_input}
	)
	list(APPEND SPIRV_BINARY_FILES ${spirv_output})
endforeach(HLSL_SOURCE)

add_custom_target(shaders DEPENDS ${SPIRV_BINARY_FILES})
