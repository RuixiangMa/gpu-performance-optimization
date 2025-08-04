# GPU Performance Optimization Algorithms

This project collects and implements various GPU performance optimization algorithms, aiming to enhance GPU computing efficiency, reduce memory bandwidth consumption, and improve overall execution speed.

## Table of Contents

* [Introduction](#introduction)
* [Project Structure](#project-structure)
* [Optimization Algorithms](#optimization-algorithms)
* [Memory Optimization Examples](#memory-optimization-examples)

## Introduction

With the development of LLMs and high-performance computing, GPUs play an increasingly important role in accelerating computational tasks. However, to fully leverage GPU potential, various optimization techniques are needed to improve performance. This project serves as a collection of GPU optimization techniques and algorithms that can be applied to enhance computational efficiency.

## Project Structure

The project is organized to demonstrate various GPU optimization concepts:
- `Memory/` - Contains memory access pattern optimizations
  - `Global-Memory/` - Global memory access optimizations
  - `Shared-Memory/` - Shared memory optimizations

## Optimization Algorithms

| Algorithm Category | Algorithm Name | Brief Description | Status |
|-------------------|----------------|-------------------|--------|
| Memory Access | Shared Memory Optimization | Shows efficient shared memory usage for data reuse and bank conflict avoidance | ✅ |
| Memory Access | Global Memory Coalescing | Demonstrates optimal memory access patterns for better bandwidth utilization | ✅ |

## Memory Optimization Examples

This project includes practical examples demonstrating key memory optimization techniques:

### Shared Memory Optimization
- **Matrix multiplication Optimization**: Matrix multiplication optimization using shared memory data reuse
- **Stencil Computation Optimization**: 5-point stencil computation with shared memory data reuse
- **Bank Conflict Resolution**: Demonstrates shared memory bank conflicts detection and avoidance
- Includes shared memory initialization and synchronization examples
- Examples are located in the `Memory/Shared-Memory` directory.

### Global Memory Coalescing
- Demonstrates the impact of memory access patterns on performance
- Shows coalesced vs non-coalesced memory access
- Includes examples of various memory access patterns and their impact on performance:
  - Coalesced access patterns
  - Strided access patterns (2x, 4x, 8x, 16x)
  - Misaligned access patterns (+1, +4, +16, +32, +128, +256, +512 bytes)
- Examples are located in the `Memory/Global-Memory` directory.
- Detailed analysis of memory access patterns in `Memory/Global-Memory/README.md`

## Building and Running Examples

To build and run the examples, navigate to the respective directories and use the provided Makefiles:

```bash
# For Global Memory examples
cd Memory/Global-Memory
make all
./aligned_memory_optimization

# For Shared Memory examples
cd Memory/Shared-Memory
make all
./matrix_multiplication_optimization
./stencil_optimization
./bank_conflict
```

## References

1. Hijma, Pieter and Heldens, Stijn and Sclocco, Alessio and van Werkhoven, Ben and Bal, Henri E. Optimization Techniques for GPU Programming. Association for Computing Machinery, 2023.
