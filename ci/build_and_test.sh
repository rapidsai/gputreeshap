#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
set -euo pipefail
. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate C++ build & test dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key build \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Build C++ targets"
./build.sh

rapids-logger "GoogleTest"
./build/TestGPUTreeShap

rapids-logger "Run Example"
./build/GPUTreeShapExample
