# Write a Makefile for the cutlass_learn directory
# The cutlass repo is in the external directory: ../external/cutlass/include

# Set the CUTLASS_REPO_PATH to the path of the cutlass repo
CUTLASS_REPO_PATH = /home/jk/transformerengine/3rdparty/cutlass
CUTLASS_HEADERS = ${CUTLASS_REPO_PATH}/include 
CUTLASS_TOOL_HEADERS = ${CUTLASS_REPO_PATH}/tools/util/include
CUTLASS_TEST_HEADERS = ${CUTLASS_REPO_PATH}/test/unit/common
CUTLASS_EXAMPLE_HEADERS = ${CUTLASS_REPO_PATH}/examples/common
CXX_FLAGS=--expt-relaxed-constexpr -std=c++17
HOST_DEBUG_FLAGS=-g -O0
DEBUG_FLAGS=${HOST_DEBUG_FLAGS} -G  #-lineinfo
CUTLASS_DEBUG_FLAG=-DCUTLASS_DEBUG_TRACE_LEVEL=2
PROFILE_FLAGS=-lineinfo -g
CUDA_ARCH=compute_100a
CUDA_CODE=sm_100a
ARCH_FLAGS=--generate-code=arch=${CUDA_ARCH},code=${CUDA_CODE}
CUTE_DEBUG=-DDEBUG_CUTE

.PHONY: all clean tutorial

all: clean rht_gemm

tutorial: clean epilogue_tutorial

rht_gemm: rht_gemm.cu
	nvcc ${HOST_DEBUG_FLAGS} ${ARCH_FLAGS} ${CUTLASS_DEBUG_FLAG} ${CXX_FLAGS} -I${CUTLASS_HEADERS} -o $@ $<

epilogue_tutorial: tutorial-05.cu
	nvcc ${HOST_DEBUG_FLAGS} ${ARCH_FLAGS} ${CUTLASS_DEBUG_FLAG} ${CXX_FLAGS} -I${CUTLASS_HEADERS} -I${CUTLASS_TOOL_HEADERS} -I${CUTLASS_EXAMPLE_HEADERS} -o $@ $<

clean:
	rm -rf rht_gemm epilogue_tutorial