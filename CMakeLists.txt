#=============================================================================
# Copyright (c) 2020-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)


include(FetchContent)

project(GPUTreeShap VERSION 24.10.00 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include(fetch_rapids.cmake)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(CUML)

option(BUILD_GTEST "Build google tests" OFF)
option(BUILD_EXAMPLES "Build examples" OFF)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)

# Use release build by default
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

# add third party dependencies using CPM
rapids_cpm_init()

# GPUTreeSHAP target is header-only
add_library(GPUTreeShap INTERFACE)
add_library(GPUTreeShap::GPUTreeShap ALIAS GPUTreeShap)
target_sources(GPUTreeShap INTERFACE ${GPUTreeShap_SOURCE_DIR}/GPUTreeShap/gpu_treeshap.h)
target_include_directories(GPUTreeShap INTERFACE ${GPUTreeShap_SOURCE_DIR})

set_property(TARGET GPUTreeShap PROPERTY CXX_STANDARD 17)
set_property(TARGET GPUTreeShap PROPERTY CUDA_STANDARD 17)
set(COMPILE_OPTIONS --expt-extended-lambda -lineinfo --Werror all-warnings)
set(GCC_COMPILE_OPTIONS -Xcompiler -Werror,-Wall,-Wextra)

if(BUILD_GTEST)
  include(${rapids-cmake-dir}/cpm/gtest.cmake)
  rapids_cpm_gtest()

  add_executable(TestGPUTreeShap tests/test_gpu_treeshap.cu)
  target_link_libraries(TestGPUTreeShap PRIVATE GPUTreeShap)
  target_compile_options(TestGPUTreeShap PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: ${COMPILE_OPTIONS}>)
  if(NOT MSVC)
    target_compile_options(TestGPUTreeShap PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${GCC_COMPILE_OPTIONS}>)
  endif()
  target_link_libraries(TestGPUTreeShap PRIVATE GTest::gtest GTest::gtest_main)
endif()

if(BUILD_EXAMPLES)
  add_executable(GPUTreeShapExample example/example.cu)
  target_link_libraries(GPUTreeShapExample PRIVATE GPUTreeShap)
  target_compile_options(GPUTreeShapExample PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${COMPILE_OPTIONS}>)
  if(NOT MSVC)
    target_compile_options(GPUTreeShapExample PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${GCC_COMPILE_OPTIONS}>)
  endif()
endif()

if(BUILD_BENCHMARKS)
  include(${rapids-cmake-dir}/cpm/gbench.cmake)
  rapids_cpm_gbench()

  add_executable(BenchmarkGPUTreeShap benchmark/benchmark_gpu_treeshap.cu)
  target_link_libraries(BenchmarkGPUTreeShap PRIVATE GPUTreeShap)
  target_compile_options(BenchmarkGPUTreeShap PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: ${COMPILE_OPTIONS}>)
  if(NOT MSVC)
    target_compile_options(BenchmarkGPUTreeShap PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${GCC_COMPILE_OPTIONS}>)
  endif()
  target_link_libraries(BenchmarkGPUTreeShap  PRIVATE benchmark::benchmark)
endif()

include(cmake/doxygen.cmake)
add_doxygen_target(IN_DOXYFILE Doxyfile.in
  OUT_DOXYFILE ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
  CWD ${CMAKE_CURRENT_BINARY_DIR})
