# Shared Memory Optimization

This repository contains CUDA implementations for shared memory optimization techniques, focusing on data reuse and bank conflict resolution.

## Completed Algorithms

### Stencil Computation Optimization
- **Algorithm**: 5-point stencil computation with shared memory data reuse
- **File**: `stencil_optimization.cu`
- **Description**: Demonstrates performance improvements achieved through shared memory data reuse in stencil computations commonly used in image processing and PDE solvers
- **Features**:
  - Global memory baseline implementation
  - Shared memory optimized implementation
  - Performance comparison with detailed metrics
  - Bandwidth calculation and analysis
  - Verification of result correctness

### Matrix multiplication Optimization
- **Algorithm**: Matrix multiplication with shared memory data reuse
- **File**: `matrix_multiplication_optimization.cu`
- **Description**: Demonstrates performance improvements achieved through shared memory data reuse in matrix multiplication
- **Features**:
  - Global memory baseline implementation
  - Shared memory optimized implementation
  - Performance comparison with detailed metrics
  - Bandwidth calculation and analysis
  - Verification of result correctness

### Bank Conflict Resolution
- **Algorithm**: Shared memory bank conflict detection and resolution
- **File**: `bank_conflict.cu`
- **Description**: Implements techniques to identify and mitigate shared memory bank conflicts that can severely impact GPU performance
- **Features**:
  - Bank conflict detection
  - Padding strategies 
  - Data reordering techniques 
  - Swizzling techniques 
  - Performance comparison between different approaches

## Build Instructions

```bash
# Build stencil optimization example
make stencil_optimization

# Build data reuse optimization example
make Matrix_multiplication_optimization

# Build bank conflict resolution example
make bank_conflict
```

## Usage

```bash
# Run stencil optimization example
./stencil_optimization

# Run data reuse optimization example
./Matrix_multiplication_optimization

# Run bank conflict resolution example
./bank_conflict
```
