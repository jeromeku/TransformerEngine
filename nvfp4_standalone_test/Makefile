# 1. Use the shell function to execute realpath
# We use := (simply expanded variables) so the shell runs only once
CUDNN_PATH := $(shell realpath -L ../.venv/lib/python3.12/site-packages/nvidia/cudnn)
CUDNN_HOME := $(CUDNN_PATH)
CUDA_HOME  := /home/jeromeku/cuda

# # Handle the conditional logic for CPATH (appending only if it exists)
# ifdef CPATH
#     export CPATH := $(CUDNN_HOME)/include:$(CPATH)
# else
#     export CPATH := $(CUDNN_HOME)/include
# endif

# 3. Export the other variables so they are available to shell commands in recipes
export CUDNN_HOME
export CUDNN_PATH
export CUDA_HOME

# Example Target
TE_ROOT ?= ..
TE_LIB_DIR ?= $(TE_ROOT)

NVCC ?= nvcc
CUDA_ARCH ?= 100

BUILD_DIR ?= build
BIN := $(BUILD_DIR)/test_cast_nvfp4_transpose

OPENMP ?= 1

# cuDNN discovery:
# - If cuDNN headers/libs aren't in default system locations, set one of:
#     CUDNN_INCLUDE_DIR=/path/to/cudnn/include
#     CUDNN_LIB_DIR=/path/to/cudnn/lib64
# - Or use env vars commonly used by TE CMake:
#     CUDNN_INCLUDE_PATH, CUDNN_LIBRARY_PATH, CUDNN_PATH


CUDNN_INCLUDE_DIR := $(CUDNN_PATH)/include
CUDNN_LIB_DIR := $(CUDNN_PATH)/lib
CUDNN_SO := $(firstword $(wildcard $(CUDNN_LIB_DIR)/libcudnn.so) $(wildcard $(CUDNN_LIB_DIR)/libcudnn.so.*))

SRCS := \
  $(TE_ROOT)/tests/cpp/test_common.cu \
  test_cast_nvfp4_transpose.cu

INCLUDES := \
  -Iinclude \
  -I$(TE_ROOT)/tests/cpp \
  -I$(TE_ROOT)/transformer_engine/common/include \
  -I$(TE_ROOT)/transformer_engine/common \
  -I$(TE_ROOT)/transformer_engine

ifneq ($(strip $(CUDNN_INCLUDE_DIR)),)
  INCLUDES += -I$(CUDNN_INCLUDE_DIR)
endif

GENCODE ?= -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)

NVCCFLAGS ?= -O0 -g -std=c++17 $(GENCODE) $(INCLUDES) --expt-extended-lambda --expt-relaxed-constexpr
ifeq ($(OPENMP),1)
  NVCCFLAGS += -Xcompiler -fopenmp
endif

NVCC_LDFLAGS ?= -L$(TE_LIB_DIR) -Xlinker -rpath -Xlinker $(abspath $(TE_LIB_DIR))

ifneq ($(strip $(CUDNN_LIB_DIR)),)
  NVCC_LDFLAGS += -L$(CUDNN_LIB_DIR) -Xlinker -rpath -Xlinker $(abspath $(CUDNN_LIB_DIR))
endif

LIBS ?= -ltransformer_engine -lcudart -lnvrtc
ifeq ($(strip $(CUDNN_SO)),)
  LIBS += -lcudnn
else
  LIBS += -l:$(notdir $(CUDNN_SO))
endif

.PHONY: all clean run check-env test_paths

test_paths:
	@echo "--- Environment Check ---"
	@echo "CUDNN_PATH: $(CUDNN_PATH)"
	@echo "CUDA_HOME:  $(CUDA_HOME)"
	@echo "PATH:       $(PATH)"
	@echo "CPATH:      $(CPATH)"
  
all: $(BIN)

check-env:
	@if [ ! -f "$(TE_LIB_DIR)/libtransformer_engine.so" ]; then \
	  echo "NOTE: libtransformer_engine.so not found in TE_LIB_DIR=$(TE_LIB_DIR) (override TE_LIB_DIR if needed)."; \
	fi
	@if [ -n "$(strip $(CUDNN_INCLUDE_DIR))" ] && [ ! -f "$(CUDNN_INCLUDE_DIR)/cudnn.h" ]; then \
	  echo "ERROR: CUDNN_INCLUDE_DIR is set but cudnn.h not found at $(CUDNN_INCLUDE_DIR)/cudnn.h"; \
	  exit 2; \
	fi
	@if [ -n "$(strip $(CUDNN_LIB_DIR))" ] && [ -z "$(strip $(CUDNN_SO))" ]; then \
	  echo "ERROR: cuDNN library not found under CUDNN_LIB_DIR=$(CUDNN_LIB_DIR) (expected libcudnn.so or libcudnn.so.*)."; \
	  exit 2; \
	fi
	@if [ -z "$(strip $(CUDNN_INCLUDE_DIR))" ]; then \
	  echo "NOTE: cudnn.h must be discoverable via default includes, or set CUDNN_INCLUDE_DIR/CUDNN_INCLUDE_PATH/CUDNN_PATH."; \
	fi

$(BIN): check-env $(SRCS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(SRCS) -o $@ $(NVCC_LDFLAGS) $(LIBS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

run: $(BIN)
	LD_LIBRARY_PATH="$(abspath $(TE_LIB_DIR)):$$LD_LIBRARY_PATH" $(BIN) --shape 128x128 --dtype bf16 --activation identity

clean:
	rm -rf $(BUILD_DIR)
