#/bin/bash

set -euo pipefail

# uv pip install cmake ninja pybind11 wheel setuptools
# uv pip install torch --torch-backend=cu129
# uv pip install nvidia-mathdx==25.1.1
# uv pip install nvshmem4py-cu12


REPO_ROOT=`realpath -L .`
export CUDNN_HOME=`realpath -L .venv/lib/python3.12/site-packages/nvidia/cudnn`
export CUDNN_PATH=`realpath -L .venv/lib/python3.12/site-packages/nvidia/cudnn`
export CUDA_HOME=/home/jeromeku/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export CUDNN_PATH=`realpath -L .venv/lib/python3.12/site-packages/nvidia/cudnn`
export CUDNN_HOME=`realpath -L .venv/lib/python3.12/site-packages/nvidia/cudnn`
export CPATH="$CUDNN_HOME/include${CPATH:+:$CPATH}"

export NVTE_BUILD_DEBUG=1

export NVTE_CMAKE_EXTRA_ARGS="-DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DCMAKE_VERBOSE_MAKEFILE=1 \
-DCMAKE_EXPORT_COMPILE_COMMANDS=1"

export NVTE_FRAMEWORK="pytorch"
export NVTE_ENABLE_NVSHMEM=0
export NVTE_CUDA_ARCHS="100a"

uv pip install --no-build-isolation -v --editable ".[pytorch]" 2>&1 | tee _build.log