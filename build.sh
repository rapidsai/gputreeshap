#!/bin/bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION.
set -euo pipefail

cmake -B build -S . \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGPUTREESHAP_BUILD_GTEST=ON \
    -DGPUTREESHAP_BUILD_EXAMPLES=ON \
    -DGPUTREESHAP_BUILD_BENCHMARKS=ON

cmake --build build -j4

RAPIDS_VERSION_MAJOR_MINOR=$(cat ./RAPIDS_VERSION) \
    cmake --build build --target docs_gputreeshap
