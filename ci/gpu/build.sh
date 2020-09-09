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
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4
export CUDA_REL=${CUDA_VERSION%.*}

# Set home to the job's workspace
export HOME=$WORKSPACE


################################################################################
# SETUP - Check environment
################################################################################

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
