# Global Memory Optimization

This repository contains CUDA implementations for global memory optimization techniques, specifically focusing on coalesced memory access to improve GPU performance.

## Completed Algorithms

### Memory Access Pattern Comparison
- **Algorithm**: Comprehensive analysis of different memory access patterns
- **File**: `aligned_memory_optimization.cu`
- **Description**: Demonstrates the impact of various memory access patterns on global memory performance including coalesced access, strided access, misaligned access, boundary crossing, and random access.
- **Features**:
  - Multiple access patterns: coalesced, strided (2x/4x), misaligned (+1 float/+4 bytes/+16 bytes/+32 bytes), boundary crossing, and random access
  - Performance metrics collection with timing and bandwidth calculations
  - Detailed performance comparison and bandwidth efficiency analysis
  - Explanation of results and impact of compiler optimizations

### Matrix Transpose Optimization
- **Algorithm**: Matrix transpose with coalesced vs uncoalesced memory access
- **File**: `coalesced_memory_optimazation.cu`
- **Description**: Compares naive matrix transpose (uncoalesced memory access) with optimized version using shared memory and coalesced access patterns.
- **Features**:
  - Naive transpose implementation with uncoalesced memory writes
  - Optimized transpose using shared memory and coalesced access patterns
  - Performance comparison between both approaches
  - Result verification

## Build Instructions

```bash
make
```

## Run Instructions

```bash
# Run memory access pattern comparison
./aligned_memory_optimization

# Run matrix transpose optimization comparison
./coalesced_memory_optimazation
```
## Querying GPU Memory Properties

To understand the memory characteristics of your GPU, you can use the `query_gpu_properties.cu` program:

```bash
nvcc -o query_gpu_properties query_gpu_properties.cu
./query_gpu_properties
```

This program will display important GPU properties including:
- Compute capability
- Memory clock rate
- Memory bus width
- L2 cache size
- Memory access alignment requirements

Understanding your GPU's memory access granularity is crucial for optimizing memory access patterns.
