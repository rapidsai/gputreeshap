#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION.
#########################################
# GPUTreeShap GPU build and test script for CI #
#########################################

set -e
NUMARGS=$#
ARGS=$*

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Set path and build parallel level
export PATH=/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4
export CUDA_REL=${CUDA_VERSION%.*}

# Set home to the job's workspace
export HOME=$WORKSPACE


################################################################################
# SETUP - Check environment
################################################################################

logger "Install cmake..."
mkdir cmake
cd cmake
wget https://github.com/Kitware/CMake/releases/download/v3.18.2/cmake-3.18.2-Linux-x86_64.sh
sh cmake-3.18.2-Linux-x86_64.sh --skip-license
export PATH=$PATH:$PWD/bin
cd ..

logger "Install gtest..."
wget https://github.com/google/googletest/archive/release-1.10.0.zip
unzip release-1.10.0.zip
mv googletest-release-1.10.0 gtest && cd gtest
cmake . && make
cp -r googletest/include include
export GTEST_ROOT=$PWD
cd ..

logger "Check environment..."
env


logger "Check GPU usage..."
nvidia-smi

$CC --version
$CXX --version

################################################################################
# BUILD - Build tests
################################################################################

logger "Build C++ targets..."
mkdir $WORKSPACE/build
cd $WORKSPACE/build
cmake .. -DBUILD_GTEST=ON -DBUILD_EXAMPLES=ON
make -j

################################################################################
# TEST - Run GoogleTest
################################################################################

logger "GoogleTest..."
cd $WORKSPACE/build
./TestGPUTreeShap

################################################################################
# Run example
################################################################################
logger "Example..."
cd $WORKSPACE/build
./GPUTreeShapExample
