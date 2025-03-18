#!/bin/bash
# Copyright (c) 2022-2025, NVIDIA CORPORATION.
set -euo pipefail

mkdir -p build
cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGPUTREESHAP_BUILD_GTEST=ON \
    -DGPUTREESHAP_BUILD_EXAMPLES=ON \
    -DGPUTREESHAP_BUILD_BENCHMARKS=ON
make -j4
cmake --build . --target docs_gputreeshap
