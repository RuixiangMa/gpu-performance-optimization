# Makefile for CUDA Bank Conflict Test

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
all: bank_conflict

# Linking rules
bank_conflict: bank_conflict.cu
	$(NVCC) $(CFLAGS) -o bank_conflict bank_conflict.cu

# Clean up
clean:
	rm -f bank_conflict

.PHONY: all clean
