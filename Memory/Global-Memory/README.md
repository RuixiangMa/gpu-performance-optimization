# Global Memory Optimization

This repository contains CUDA implementations for global memory optimization techniques, specifically focusing on coalesced memory access to improve GPU performance.

## Completed Algorithms

### Coalesced Memory Access
- **Algorithm**: Global memory coalesced access demonstration
- **File**: `global_memory_coalesced.cu`
- **Description**: Implements techniques to demonstrate the impact of coalesced and uncoalesced memory access on global memory performance.
- **Features**:
  - Comparison between coalesced and uncoalesced memory access
  - Performance metrics collection
  - Visualization of performance differences

## Build Instructions

```bash
make global_memory_optimization
```
## Run Instructions
```bash
./global_memory_coalesced
```