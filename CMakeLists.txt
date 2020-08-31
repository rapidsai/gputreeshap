cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(GPUTreeShap LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

option(BUILD_GTEST "Build google tests" OFF)
option(BUILD_EXAMPLES "Build examples" OFF)

# GPUTreeSHAP target is header-only
add_library(GPUTreeShap INTERFACE)
add_library(GPUTreeShap::GPUTreeShap ALIAS GPUTreeShap)
target_sources(GPUTreeShap INTERFACE ${GPUTreeShap_SOURCE_DIR}/GPUTreeShap/gpu_treeshap.h)
target_include_directories(GPUTreeShap INTERFACE ${GPUTreeShap_SOURCE_DIR})

if(BUILD_GTEST)
  find_package(GTest REQUIRED)
  add_executable(TestGPUTreeShap tests/test_gpu_treeshap.cu)
  target_link_libraries(TestGPUTreeShap PRIVATE GPUTreeShap)
  target_compile_options(TestGPUTreeShap PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda>)
  target_link_libraries(TestGPUTreeShap  PRIVATE GTest::GTest GTest::Main)
  add_custom_command(TARGET TestGPUTreeShap POST_BUILD COMMAND cpplint ${GPUTreeShap_SOURCE_DIR}/GPUTreeShap/gpu_treeshap.h ${GPUTreeShap_SOURCE_DIR}/tests/test_gpu_treeshap.cu ${GPUTreeShap_SOURCE_DIR}/example/example.cu)
endif()

if(BUILD_EXAMPLES)
  add_executable(GPUTreeShapExample example/example.cu)
  target_link_libraries(GPUTreeShapExample PRIVATE GPUTreeShap)
endif()
