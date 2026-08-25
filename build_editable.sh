#!/bin/bash

CUDNN_PATH=$(python -c 'import nvidia.cudnn as cudnn; print(cudnn.__path__[0])')

export CUDNN_PATH
export CUDNN_HOME="${CUDNN_PATH}"
export CPATH="${CUDNN_PATH}/include:${CPATH:-}"
export LIBRARY_PATH="${CUDNN_PATH}/lib:${LIBRARY_PATH:-}"
export NVTE_BUILD_THREADS_PER_JOB=8
export NVTE_FRAMEWORK=pytorch
export NVTE_CUDA_ARCHS="100"
export NVTE_CUDA_INCLUDE_DIR=/usr/local/cuda
export NVTE_CMAKE_BUILD_DIR=build
echo $LIBRARY_PATH $CUDNN_PATH $CPATH
uv pip install --no-build-isolation --no-deps -v --editable . 2>&1 | tee te.build.log