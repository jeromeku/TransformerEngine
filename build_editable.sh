#/bin/bash

set -euo pipefail

# uv pip install cmake ninja pybind11 wheel setuptools
# uv pip install torch --torch-backend=cu129
# uv pip install nvidia-mathdx==25.1.1
# uv pip install nvshmem4py-cu12

# rm -rf build

REPO_ROOT=`realpath -L .`

export CUDNN_HOME=`realpath -L .venv/lib/python3.12/site-packages/nvidia/cudnn`
export CUDNN_PATH=`realpath -L .venv/lib/python3.12/site-packages/nvidia/cudnn`
# export NVSHMEM_HOME=`realpath -L .venv/lib/python3.12/site-packages/nvidia/nvshmem`
# echo $NVSHMEM_HOME
# EXTRA_INCLUDES=" -I${REPO_ROOT}/transformer_engine/common/include"
# export NVTE_CMAKE_BUILD_DIR="build/dev"
# export LD_LIBRARY_PATH=${NVSHMEM_HOME}/lib

echo $CUDNN_HOME
echo $REPO_ROOT
export CPATH="$CUDNN_HOME/include${CPATH:+:$CPATH}"

# CMAKE Flags
DEBUG_FLAGS="-g3 -O0"
# export NVTE_BUILD_DEBUG=1
# "-DCMAKE_CXX_FLAGS=${DEBUG_FLAGS}" \
CUDA_DEBUG_FLAGS="-O0"
export CUDAFLAGS="--Ofast-compile=max -g"

CMAKE_EXTRA_ARGS=( -DCMAKE_VERBOSE_MAKEFILE=1 -DCMAKE_EXPORT_COMPILE_COMMANDS=1)
CMAKE_EXTRA_ARGS+=(-DCMAKE_CUDA_FLAGS_DEBUG=${CUDA_DEBUG_FLAGS})
# CMAKE_EXTRA_ARGS+=(-DCMAKE_CUDA_FLAGS=${CUDAFLAGS})
CMAKE_EXTRA_ARGS+=(-DCMAKE_CUDA_FLAGS_RELEASE=${CUDA_DEBUG_FLAGS})

# Easiest way is to hardcode set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -g --Ofast-compile=max") in CMakeLIsts.txt

export NVTE_CMAKE_EXTRA_ARGS="${CMAKE_EXTRA_ARGS[@]}"
# export NVTE_BUILD_THREADS_PER_JOB=8

export CXXFLAGS=${DEBUG_FLAGS}

export CC=clang
export CXX=clang++

# TE Flags
export NVTE_FRAMEWORK="pytorch"
export NVTE_ENABLE_NVSHMEM=0
export NVTE_CUDA_ARCHS="100a"

uv pip install --no-build-isolation -v --editable ".[pytorch]" 2>&1 | tee _build.log
