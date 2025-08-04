# Makefile for GPU memory optimization examples

NVCC ?= nvcc
CFLAGS ?= -O3 -arch=sm_89

# Default target - builds all programs
all: bank_conflict matrix_multiplication_optimization reduce_optimization stencil_optimization

# Compile bank_conflict program
bank_conflict: bank_conflict.cu
	$(NVCC) $(CFLAGS) -o bank_conflict bank_conflict.cu

# Compile matrix_multiplication_optimization program
matrix_multiplication_optimization: matrix_multiplication_optimization.cu
	$(NVCC) $(CFLAGS) -o matrix_multiplication_optimization matrix_multiplication_optimization.cu

# Compile reduce_optimization program
reduce_optimization: reduce_optimization.cu
	$(NVCC) $(CFLAGS) -o reduce_optimization reduce_optimization.cu

# Compile stencil_optimization program
stencil_optimization: stencil_optimization.cu
	$(NVCC) $(CFLAGS) -o stencil_optimization stencil_optimization.cu

# Clean build artifacts
clean:
	rm -f bank_conflict matrix_multiplication_optimization reduce_optimization stencil_optimization

.PHONY: all clean