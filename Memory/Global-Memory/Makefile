# Makefile for CUDA Global Memory Optimization Test

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

# Common flags
CFLAGS = -O3 -arch=sm_89

# Default target
all: global_memory_optimization

# Linking rules
global_memory_optimization: global_memory_optimization.cu
	$(NVCC) $(CFLAGS) -o global_memory_optimization global_memory_optimization.cu

# Clean up
clean:
	rm -f global_memory_optimization

.PHONY: all clean