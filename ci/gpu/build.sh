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

logger "Check environment..."
env


logger "Check GPU usage..."
nvidia-smi

$CC --version
$CXX --version

# Update git if on centos
if [ -f /etc/redhat-release ]; then
  yum remove git*
  yum -y install https://packages.endpoint.com/rhel/7/os/x86_64/endpoint-repo-1.7-1.x86_64.rpm
  yum -y install git
  git --version
fi

export PATH=/conda/bin:$PATH

################################################################################
# BUILD - Build tests
################################################################################

logger "Build C++ targets..."
mkdir $WORKSPACE/build
cd $WORKSPACE/build
cmake .. -DBUILD_GTEST=ON -DBUILD_EXAMPLES=ON -DBUILD_BENCHMARKS=ON
make -j

################################################################################
# TEST - Run GoogleTest
################################################################################

logger "GoogleTest..."
cd $WORKSPACE/build
./TestGPUTreeShap

################################################################################
# TEST - Run Benchmarks
################################################################################
logger "Benchmark..."
cd $WORKSPACE/build
./BenchmarkGPUTreeShap --benchmark_out=gputreeshap_bench.json --benchmark_out_format=json
curl -L https://raw.githubusercontent.com/rapidsai/benchmark/main/parser/GBenchToASV.py --output GBenchToASV.py
python GBenchToASV.py -d . -t ${S3_ASV_DIR} -n gputreeshap -b $(git rev-parse --abbrev-ref HEAD)

################################################################################
# Run example
################################################################################
logger "Example..."
cd $WORKSPACE/build
./GPUTreeShapExample
