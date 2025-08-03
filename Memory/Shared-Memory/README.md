# Shared Memory Optimization

This repository contains CUDA implementations for shared memory optimization techniques, specifically focusing on bank conflict resolution through data reordering and swizzling.

## Completed Algorithms

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
make
```

## Usage

```bash
./bank_conflict
```
