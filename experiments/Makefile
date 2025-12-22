# -----------------------------
# Local overrides (optional)
# -----------------------------
# Put machine-specific paths here (won't conflict with git).
-include config.mk

# -----------------------------
# Toolchain + paths
# -----------------------------
CUDA_HOME       ?= /home/jeromeku/cuda
NVCC            ?= $(CUDA_HOME)/bin/nvcc

# Use clang as host compiler (override: make HOST_COMPILER=/path/to/g++)
HOST_COMPILER   ?= g++ #clang++
NVCC_HOST       := -ccbin $(HOST_COMPILER)

CUTLASS_ROOT     ?= /home/jeromeku/transformerengine/3rdparty/cutlass

# -----------------------------
# Targets / sources
# -----------------------------
BIN_DIR         ?= bin

PROGRAMS        := epilogue_tutorial rht_gemm stochastic_rounding multicast_tutorial 2sm_tutorial tma_test wgmma_tma_sm90

SRC_epilogue_tutorial     := tutorial-05.cu
SRC_multicast_tutorial    := multicast_tutorial.cu
SRC_2sm_tutorial          := 2sm_tutorial.cu
SRC_rht_gemm              := rht_gemm.cu
SRC_stochastic_rounding   := stochastic_rounding.cu
SRC_tma_test			  := tma_test.cu
SRC_wgmma_tma_sm90        := wgmma_tma_sm90.cu
BIN_TARGETS := $(addprefix $(BIN_DIR)/,$(PROGRAMS))

# -----------------------------
# Includes
# -----------------------------
INC_DIRS := \
  $(CUTLASS_ROOT)/include \
  $(CUTLASS_ROOT)/tools/util/include \
  $(CUTLASS_ROOT)/examples/common

# Add more include roots like this:
# INC_DIRS += /path/to/other/headers

INCLUDES := $(addprefix -I,$(INC_DIRS))

# -----------------------------
# CUDA arch
# -----------------------------
# Override like: make SM=90a
SM              ?= 100a
GENCODE         := --generate-code=arch=compute_$(SM),code=sm_$(SM)

# -----------------------------
# Feature toggles (set on make cmdline)
# -----------------------------
# DEBUG=1: host+device debug, -O0, frame pointers, etc.
# TRACE=1: compile-time traces (nvcc device time trace + clang time trace)
# VERBOSE=1: nvcc -v, resource usage, ptxas -v
# CUTE_DEBUG=1: -DDEBUG_CUTE
DEBUG           ?= 1
TRACE           ?= 0
VERBOSE         ?= 0
CUTE_DEBUG      ?= 0

# CUTLASS debug trace
CUTLASS_DEBUG_TRACE_LEVEL ?= 2

# -----------------------------
# Flag buckets (keep them unquoted)
# -----------------------------
CXX_STDFLAGS    := --expt-relaxed-constexpr -std=c++17

# NVCC-level flags (nvcc understands these)
NVCCFLAGS_BASE  := $(CXX_STDFLAGS) $(GENCODE)
DEFINES_BASE    := -DCUTLASS_DEBUG_TRACE_LEVEL=$(CUTLASS_DEBUG_TRACE_LEVEL)

# Device debug/trace flags
NVCCFLAGS_DEBUG := -g -O0 -lineinfo
NVCCFLAGS_TRACE := -fdevice-time-trace=-

# Host compiler flags (will be forwarded via -Xcompiler)
HOST_CXXFLAGS_DEBUG := -g3 -O0 -fno-omit-frame-pointer -fno-optimize-sibling-calls -fno-inline
HOST_CXXFLAGS_TRACE := -ftime-trace

# ptxas flags (forwarded via -Xptxas)
PTXASFLAGS_VERBOSE := -v

# nvcc build verbosity / reporting
NVCCFLAGS_VERBOSE := -v --resource-usage

# Optional cute debug define
ifeq ($(CUTE_DEBUG),1)
  DEFINES_BASE += -DDEBUG_CUTE
endif

# -----------------------------
# Helpers: forward flags safely (no quoting footguns)
# -----------------------------
# Usage: $(call XCOMPILER, $(HOST_CXXFLAGS_DEBUG))
XCOMPILER = $(foreach f,$(1),-Xcompiler $(f))
XPTXAS    = $(foreach f,$(1),-Xptxas=$(f))

# -----------------------------
# Assemble final flags from toggles
# -----------------------------
NVCCFLAGS := $(NVCCFLAGS_BASE)
DEFINES   := $(DEFINES_BASE)

HOST_CXXFLAGS :=
PTXASFLAGS    :=

ifeq ($(DEBUG),1)
  NVCCFLAGS     += $(NVCCFLAGS_DEBUG)
  HOST_CXXFLAGS += $(HOST_CXXFLAGS_DEBUG)
endif

ifeq ($(TRACE),1)
  NVCCFLAGS += $(NVCCFLAGS_TRACE)
  # Only add -ftime-trace if using clang-ish host compiler
  ifneq (,$(findstring clang,$(notdir $(HOST_COMPILER))))
    HOST_CXXFLAGS += $(HOST_CXXFLAGS_TRACE)
  endif
endif

ifeq ($(VERBOSE),1)
  NVCCFLAGS  += $(NVCCFLAGS_VERBOSE)
  PTXASFLAGS += $(PTXASFLAGS_VERBOSE)
endif

# -----------------------------
# Phonies
# -----------------------------
.PHONY: all clean print-vars
all: $(BIN_TARGETS)

# Ensure output dir exists
$(BIN_DIR):
	@mkdir -p $@

# Generic rule: build each program from its mapped source
# (Explicit per-program deps below are clearer than clever pattern magic.)
$(BIN_DIR)/epilogue_tutorial: $(SRC_epilogue_tutorial) | $(BIN_DIR)
	$(NVCC) $(NVCC_HOST) $(NVCCFLAGS) $(DEFINES) $(INCLUDES) $(call XCOMPILER,$(HOST_CXXFLAGS)) $(call XPTXAS,$(PTXASFLAGS)) -o $@ $<

$(BIN_DIR)/multicast_tutorial: $(SRC_multicast_tutorial) | $(BIN_DIR)
	$(NVCC) $(NVCC_HOST) $(NVCCFLAGS) $(DEFINES) $(INCLUDES) $(call XCOMPILER,$(HOST_CXXFLAGS)) $(call XPTXAS,$(PTXASFLAGS)) -o $@ $<

$(BIN_DIR)/2sm_tutorial: $(SRC_2sm_tutorial) | $(BIN_DIR)
	$(NVCC) $(NVCC_HOST) $(NVCCFLAGS) $(DEFINES) $(INCLUDES) $(call XCOMPILER,$(HOST_CXXFLAGS)) $(call XPTXAS,$(PTXASFLAGS)) -o $@ $<

$(BIN_DIR)/rht_gemm: $(SRC_rht_gemm) | $(BIN_DIR)
	$(NVCC) $(NVCC_HOST) $(NVCCFLAGS) $(DEFINES) $(INCLUDES) $(call XCOMPILER,$(HOST_CXXFLAGS)) $(call XPTXAS,$(PTXASFLAGS)) -o $@ $<

$(BIN_DIR)/stochastic_rounding: $(SRC_stochastic_rounding) | $(BIN_DIR)
	$(NVCC) $(NVCC_HOST) $(NVCCFLAGS) $(DEFINES) $(INCLUDES) $(call XCOMPILER,$(HOST_CXXFLAGS)) $(call XPTXAS,$(PTXASFLAGS)) -o $@ $<

$(BIN_DIR)/tma_test: $(SRC_tma_test) | $(BIN_DIR)
	$(NVCC) -G $(NVCC_HOST) $(NVCCFLAGS) $(DEFINES) $(INCLUDES) $(call XCOMPILER,$(HOST_CXXFLAGS)) $(call XPTXAS,$(PTXASFLAGS)) -o $@ $<

$(BIN_DIR)/wgmma_tma_sm90: $(SRC_wgmma_tma_sm90) | $(BIN_DIR)
	$(NVCC) $(NVCC_HOST) $(NVCCFLAGS) $(DEFINES) $(INCLUDES) $(call XCOMPILER,$(HOST_CXXFLAGS)) $(call XPTXAS,$(PTXASFLAGS)) -o $@ $<

clean:
	rm -rf $(BIN_DIR)

# Debug helper: make print-vars
print-vars:
	@echo "NVCC            = $(NVCC)"
	@echo "HOST_COMPILER   = $(HOST_COMPILER)"
	@echo "NVCCFLAGS       = $(NVCCFLAGS)"
	@echo "DEFINES         = $(DEFINES)"
	@echo "HOST_CXXFLAGS   = $(HOST_CXXFLAGS)"
	@echo "PTXASFLAGS      = $(PTXASFLAGS)"
	@echo "INCLUDES        = $(INCLUDES)"
