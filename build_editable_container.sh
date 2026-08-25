#!/bin/bash

export NVTE_BUILD_THREADS_PER_JOB=8
export NVTE_FRAMEWORK=pytorch
export NVTE_CUDA_ARCHS="100"
export NVTE_CUDA_INCLUDE_DIR=/usr/local/cuda
export NVTE_CMAKE_BUILD_DIR=build
echo $LIBRARY_PATH $CUDNN_PATH $CPATH

pip install --no-build-isolation --no-deps -v --editable . 2>&1 | tee te.container.build.log
