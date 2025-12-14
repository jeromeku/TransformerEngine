# uv pip install nvidia-mathdx==25.1.1
export CUDA_HOME=/home/jeromeku/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export CUDNN_PATH=`realpath -L .venv/lib/python3.12/site-packages/nvidia/cudnn`
export CUDNN_HOME=`realpath -L .venv/lib/python3.12/site-packages/nvidia/cudnn`
export CPATH="$CUDNN_HOME/include${CPATH:+:$CPATH}"

# NVTE_CUDA_ARCHS="100a" NVTE_FRAMEWORK=pytorch uv pip install -v --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable 2>&1 | tee _source_build.log

# install torch nightly built for cu130
# uv pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu129 torch

# install TE CUDA 12 wheel
uv pip install --no-build-isolation transformer_engine[pytorch]
