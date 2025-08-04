# Makefile for CUDA Shared Memory Optimization Tests

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
all: stencil_optimization matrix_multiplication_optimization bank_conflict

# Linking rules
stencil_optimization: stencil_optimization.cu
	$(NVCC) $(CFLAGS) -o stencil_optimization stencil_optimization.cu

matrix_multiplication_optimization: matrix_multiplication_optimization.cu
	$(NVCC) $(CFLAGS) -o matrix_multiplication_optimization matrix_multiplication_optimization.cu

bank_conflict: bank_conflict.cu
	$(NVCC) $(CFLAGS) -o bank_conflict bank_conflict.cu

# Clean up
clean:
	rm -f stencil_optimization matrix_multiplication_optimization bank_conflict
