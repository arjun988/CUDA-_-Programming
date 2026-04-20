# FlashAttention Implementation Report

## Paper

- **Title:** FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
- **Authors:** Tri Dao et al.
- **Source:** `FlashAttention.pdf`

## Goal

Implement a working, exact attention forward pass inspired by the paper's core idea:

- Avoid materializing the full `N x N` score/probability matrices.
- Use tiled processing over keys/values.
- Use **online softmax** (`m`, `l`) to maintain numerical stability across tiles.

## What Was Implemented

File: `flash_attention.cu`

### 1) FlashAttention-style forward CUDA kernel

- Kernel: `flash_attention_forward_kernel`
- Inputs/outputs are laid out as contiguous `[B, H, N, D]`.
- Grid mapping: one block per `(b, h, query_row)`.
- Block loops over key/value tiles (`TILE_N=128`).
- Uses:
  - shared memory tile for `K` and `V`,
  - running max `m`,
  - running normalization `l`,
  - running weighted sum accumulator `acc`.
- Produces exact softmax attention output for each query row.

### 2) Naive CUDA baseline kernel

- Kernel: `naive_attention_forward_kernel`
- Computes exact attention row-by-row in two passes:
  - pass 1 for max,
  - pass 2 for sum/weighted output.
- Used for correctness and speed comparison.

### 3) CPU reference implementation

- Function: `attention_cpu_reference`
- Used to validate numerical correctness against a non-CUDA implementation.

### 4) Benchmark + verification harness

- Compares:
  - Flash kernel vs naive CUDA kernel (timing),
  - flash output vs naive output (max absolute error),
  - flash output vs CPU reference (max absolute error).

## Build and Run

From `Research/FlashAttention`:

```bash
nvcc -O3 -std=c++17 flash_attention.cu -o flash_attention
./flash_attention --b 1 --h 8 --n 512 --d 64 --causal 1
```

Parameters:

- `--b` batch size
- `--h` number of heads
- `--n` sequence length
- `--d` head dimension (`<= 128` in this implementation)
- `--causal` `1` for causal mask, `0` for full attention

## Relation to the Original Paper

### Implemented from paper concepts

- IO-aware tiling over `K/V`
- online softmax with recombination across tiles
- exact (not approximate) attention output
- causal/non-causal support

### Not yet implemented (full paper scope)

- backward pass with recomputation
- dropout fusion
- optimized block-level parallel decomposition used in production FlashAttention kernels
- BF16/FP16/Tensor Core paths
- block-sparse FlashAttention extension
- extensive end-to-end training benchmarks

## Complexity Notes

- Standard attention materializes `N x N` intermediates (high memory traffic).
- This implementation keeps only tile-sized `K/V` data and per-row running stats in fast memory, reducing global-memory traffic pressure.
- Time complexity remains quadratic in `N` (exact attention), but memory behavior is improved.

## Validation Checklist

- [ ] Compare correctness errors for multiple shapes:
  - `N in {128, 256, 512, 1024}`
  - `D in {32, 64, 128}`
  - `causal in {0, 1}`
- [ ] Record timing table flash vs naive.
- [ ] Confirm stable output for larger `N`.

## Suggested Next Steps

1. Add backward kernel with recomputation (`m`, `l` reuse).
2. Fuse masking and dropout into forward/backward kernel.
3. Improve block mapping (multiple query rows per block) to increase occupancy.
4. Add FP16/BF16 + Tensor Core implementation path.
5. Add block-sparse mode from the paper.
