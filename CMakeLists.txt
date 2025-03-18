#=============================================================================
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(FetchContent)

include(fetch_rapids.cmake)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(GPUTreeShap)

project(GPUTreeShap VERSION 25.04.00 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

option(GPUTREESHAP_BUILD_GTEST "Build google tests" OFF)
option(GPUTREESHAP_BUILD_EXAMPLES "Build examples" OFF)
option(GPUTREESHAP_BUILD_BENCHMARKS "Build benchmarks" OFF)

# Use release build by default
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

# add third party dependencies using CPM
rapids_cpm_init()

include(${rapids-cmake-dir}/cpm/cccl.cmake)
rapids_cpm_cccl(BUILD_EXPORT_SET gputreeshap-exports
                INSTALL_EXPORT_SET gputreeshap-exports)

# GPUTreeShap target is header-only
add_library(GPUTreeShap INTERFACE)
add_library(GPUTreeShap::GPUTreeShap ALIAS GPUTreeShap)
target_sources(GPUTreeShap INTERFACE ${GPUTreeShap_SOURCE_DIR}/GPUTreeShap/gpu_treeshap.h)
target_include_directories(GPUTreeShap INTERFACE ${GPUTreeShap_SOURCE_DIR})
target_link_libraries(GPUTreeShap INTERFACE CCCL::CCCL)

set_property(TARGET GPUTreeShap PROPERTY CXX_STANDARD 17)
set_property(TARGET GPUTreeShap PROPERTY CUDA_STANDARD 17)
set(COMPILE_OPTIONS --expt-extended-lambda -lineinfo --Werror all-warnings)
set(GCC_COMPILE_OPTIONS -Xcompiler -Werror,-Wall,-Wextra)

if(GPUTREESHAP_BUILD_GTEST)
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

if(GPUTREESHAP_BUILD_EXAMPLES)
  add_executable(GPUTreeShapExample example/example.cu)
  target_link_libraries(GPUTreeShapExample PRIVATE GPUTreeShap)
  target_compile_options(GPUTreeShapExample PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${COMPILE_OPTIONS}>)
  if(NOT MSVC)
    target_compile_options(GPUTreeShapExample PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${GCC_COMPILE_OPTIONS}>)
  endif()
endif()

if(GPUTREESHAP_BUILD_BENCHMARKS)
  include(${rapids-cmake-dir}/cpm/gbench.cmake)
  rapids_cpm_gbench()

  add_executable(BenchmarkGPUTreeShap benchmark/benchmark_gpu_treeshap.cu)
  target_link_libraries(BenchmarkGPUTreeShap PRIVATE GPUTreeShap ${CMAKE_DL_LIBS})
  target_compile_options(BenchmarkGPUTreeShap PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: ${COMPILE_OPTIONS}>)
  if(NOT MSVC)
    target_compile_options(BenchmarkGPUTreeShap PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${GCC_COMPILE_OPTIONS}>)
  endif()
  target_link_libraries(BenchmarkGPUTreeShap PRIVATE benchmark::benchmark)
endif()

include(cmake/doxygen.cmake)
add_doxygen_target(IN_DOXYFILE Doxyfile.in
  OUT_DOXYFILE ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
  CWD ${CMAKE_CURRENT_BINARY_DIR})
