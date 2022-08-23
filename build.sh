#!/bin/bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_GTEST=ON -DBUILD_EXAMPLES=ON -DBUILD_BENCHMARKS=ON
make -j4
cmake --build . --target docs_gputreeshap
