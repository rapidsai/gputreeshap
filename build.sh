#!/bin/bash
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Releas .. 
make -j4
