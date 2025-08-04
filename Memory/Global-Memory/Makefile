# Makefile for CUDA Global Memory Optimization Tests

# Detect if we're on macOS or Linux
UNAME_S := $(shell uname -s)

# Set compiler based on platform
ifeq ($(UNAME_S),Darwin)
    # macOS
    NVCC := /usr/local/cuda/bin/nvcc
else
    # Linux
    NVCC := nvcc
endif

# Common flags for optimized compilation
CFLAGS_OPT = -O3 -arch=sm_89
# Flags for unoptimized compilation to better observe memory access patterns
CFLAGS_DEBUG = -O0 -arch=sm_89

# Default target builds both optimized versions
all: aligned_memory_optimization coalesced_memory_optimazation

# Build optimized versions
aligned_memory_optimization: aligned_memory_optimization.cu
	$(NVCC) $(CFLAGS_OPT) -o aligned_memory_optimization aligned_memory_optimization.cu

coalesced_memory_optimazation: coalesced_memory_optimazation.cu
	$(NVCC) $(CFLAGS_OPT) -o coalesced_memory_optimazation coalesced_memory_optimazation.cu

# Build unoptimized versions to better observe memory access patterns
debug: aligned_memory_optimization_debug coalesced_memory_optimazation_debug

aligned_memory_optimization_debug: aligned_memory_optimization.cu
	$(NVCC) $(CFLAGS_DEBUG) -o aligned_memory_optimization_debug aligned_memory_optimization.cu

coalesced_memory_optimazation_debug: coalesced_memory_optimazation.cu
	$(NVCC) $(CFLAGS_DEBUG) -o coalesced_memory_optimazation_debug coalesced_memory_optimazation.cu

# Clean up
clean:
	rm -f aligned_memory_optimization coalesced_memory_optimazation
	rm -f aligned_memory_optimization_debug coalesced_memory_optimazation_debug

.PHONY: all debug clean
