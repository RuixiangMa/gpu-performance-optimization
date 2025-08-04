# Shared Memory Optimizations

This directory contains various GPU memory optimization techniques focusing on shared memory usage.

## Files:
- `bank_conflict.cu` - Demonstrates bank conflict detection and resolution
- `matrix_multiplication_optimization.cu` - Matrix multiplication optimization examples
- `reduce_optimization.cu` - Reduction operation optimizations comparison
- `stencil_optimization.cu` - Stencil computation optimizations

## Compilation:

To compile all programs in this directory:

```bash
make all
```

## Note on Verification:

In `reduce_optimization.cu`, you may see verification failures due to floating-point precision differences between the various reduction algorithms. This is normal because:

1. Each algorithm performs computations in different orders
2. Different memory access patterns and thread cooperation lead to slight variations in accumulated floating-point errors
3. The performance improvements measured are still accurate and meaningful
