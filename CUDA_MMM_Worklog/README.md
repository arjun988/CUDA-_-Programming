# CUDA MMM Worklog (Siboehm Rebuild)

This folder reproduces the matrix-multiplication optimization journey from:

- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)

Implemented kernels in `kernels.cuh` (launched from `mmm_worklog.cu`):

1. Naive
2. Global-memory coalesced indexing
3. Shared-memory cache blocking
4. 1D block tiling (multiple outputs per thread)
5. 2D block tiling (register tiling)
6. Vectorized global loads + transposed `As` shared layout

Also included:

- cuBLAS reference path for correctness + performance comparison
- Benchmark harness (warmup + timed iterations)
- Basic autotune mode for kernel 6 parameter candidates

## Build

```bash
nvcc -O3 -std=c++17 mmm_worklog.cu -lcublas -o mmm_worklog
```

## Run

```bash
# Run all implemented kernels + cuBLAS
./mmm_worklog --m 1024 --n 1024 --k 1024 --kernel all

# Run only one kernel
./mmm_worklog --m 2048 --n 2048 --k 2048 --kernel k6

# Autotune a few k6 configs
./mmm_worklog --m 1024 --n 1024 --k 1024 --kernel autotune
```

## Notes

- Current implementation expects dimensions divisible by tile sizes for best performance.
- Boundary guards are included for safety, but peak throughput comes from aligned, tile-friendly dimensions.
- This is educational/performance-learning code, not a full cuBLAS replacement.
