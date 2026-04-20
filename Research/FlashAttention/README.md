# FlashAttention (Research Implementation)

This folder contains a CUDA implementation inspired by:

- `FlashAttention.pdf`
- *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (Tri Dao et al.)

## Files

- `flash_attention.cu` - CUDA implementation + benchmarks + correctness checks
- `REPORT.md` - implementation report and gap analysis versus the full paper
- `FlashAttention.pdf` - original paper

## Build

```bash
nvcc -O3 -std=c++17 flash_attention.cu -o flash_attention
```

## Run

```bash
./flash_attention --b 1 --h 8 --n 512 --d 64 --causal 1
```

Arguments:

- `--b` batch size
- `--h` heads
- `--n` sequence length
- `--d` head dimension (`<= 128` currently)
- `--causal` causal mask (`1` or `0`)

## What this implements

- Exact attention forward pass
- Tiled `K/V` processing in shared memory
- Numerically stable online softmax over tiles
- Naive CUDA baseline and CPU reference for validation

For details and remaining work to fully match the paper, see `REPORT.md`.
