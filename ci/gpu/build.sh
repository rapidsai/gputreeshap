#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.
#########################################
# GPUTreeShap GPU build and test script for CI #
#########################################

set -e
NUMARGS=$#
ARGS=$*

# Set path and build parallel level
export PATH=/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}
export CUDA_REL=${CUDA_VERSION%.*}

# Set home to the job's workspace
export HOME=$WORKSPACE

# Install gpuCI tools
curl -s https://raw.githubusercontent.com/rapidsai/gpuci-tools/main/install.sh | bash
source ~/.bashrc
cd ~

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Check environment"
env


gpuci_logger "Check GPU usage"
nvidia-smi

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

################################################################################
# BUILD - Build tests
################################################################################

gpuci_logger "Build C++ targets"
./build.sh

################################################################################
# TEST - Run GoogleTest
################################################################################

gpuci_logger "GoogleTest"
cd $WORKSPACE/build
./TestGPUTreeShap

################################################################################
# Run example
################################################################################
gpuci_logger "Example"
cd $WORKSPACE/build
./GPUTreeShapExample
