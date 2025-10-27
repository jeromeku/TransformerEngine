# uv pip install cmake ninja pybind11 wheel setuptools
# uv pip install torch --torch-backend=cu128
# uv pip install nvidia-mathdx==25.1.1
# uv pip install nvshmem4py-cu12

# rm -rf build

REPO_ROOT=`realpath -L .`

export NVSHMEM_HOME=`realpath -L .venv/lib/python3.12/site-packages/nvidia/nvshmem`
export CUDNN_HOME=`realpath -L .venv/lib/python3.12/site-packages/nvidia/cudnn`
echo $NVSHMEM_HOME
echo $CUDNN_HOME
echo $REPO_ROOT
EXTRA_INCLUDES=" -I${REPO_ROOT}/transformer_engine/common/include"
# export NVTE_CMAKE_BUILD_DIR="build/dev"
export NVTE_BUILD_DEBUG=1

export NVTE_CMAKE_EXTRA_ARGS="-DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DCMAKE_VERBOSE_MAKEFILE=1 \
-DCMAKE_EXPORT_COMPILE_COMMANDS=1"

export NVTE_FRAMEWORK="pytorch"
export NVTE_ENABLE_NVSHMEM=0
export NVTE_CUDA_ARCHS="90"
export LD_LIBRARY_PATH=${NVSHMEM_HOME}/lib

uv pip install --no-build-isolation -v --editable ".[pytorch]" 2>&1 | tee _build.log
