cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(GPUTreeShap LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

find_package(GTest)
include_directories(${CMAKE_SOURCE_DIR})
add_executable(TestGPUTreeShap tests/test_gpu_treeshap.cu GPUTreeShap/gpu_treeshap.h)
target_include_directories(TestGPUTreeShap  PRIVATE ${GTEST_INCLUDE_DIRS})
if(MSVC)
    target_compile_options(TestGPUTreeShap PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /WX>)
else()
    target_compile_options(TestGPUTreeShap PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -Werror,-Wall,-Wextra>)
endif()
target_link_libraries(TestGPUTreeShap  PRIVATE  ${GTEST_BOTH_LIBRARIES})

add_custom_command(TARGET TestGPUTreeShap PRE_BUILD COMMAND cpplint ${CMAKE_SOURCE_DIR}/GPUTreeShap/gpu_treeshap.h ${CMAKE_SOURCE_DIR}/tests/test_gpu_treeshap.cu ${CMAKE_SOURCE_DIR}/example/example.cu)

add_executable(GPUTreeShapExample example/example.cu GPUTreeShap/gpu_treeshap.h)
