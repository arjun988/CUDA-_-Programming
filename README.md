# CUDA Programming [Notes](https://docs.nvidia.cn/cuda/cuda-c-programming-guide/index.html#)

Notes and examples for learning CUDA on NVIDIA GPUs. The repository is organized into four chapters that mirror a typical learning path.

---

## What is CUDA?

CUDA (Compute Unified Device Architecture) is NVIDIA’s parallel computing platform and programming model. It lets you run general-purpose code on the GPU using extensions to C/C++ and related APIs.

**Core ideas (in brief):**

- **Host vs device:** The CPU and its memory are the *host*; the GPU and its memory are the *device*. CUDA code often runs mostly on the host and launches work on the device.
- **Kernels:** Functions marked `__global__` that run on the GPU. Many *threads* execute the same kernel; each thread has an index you use to pick which data it processes.
- **SIMT model:** Single Instruction, Multiple Threads. Warps (typically 32 threads) advance together when they take the same path; divergent branches can reduce efficiency.
- **Memory hierarchy:** Registers (per thread), shared memory (per block), global memory (large, higher latency), plus caches and texture/surface memory where applicable. Performance often depends on how well you use registers and shared memory vs global traffic.
- **Grids and blocks:** You launch a *grid* of *thread blocks*. Block indices and thread indices (`blockIdx`, `threadIdx`, `blockDim`, `gridDim`) map threads to your problem domain.
- **Streams and concurrency:** CUDA streams order operations and allow overlap of copy and compute when the hardware and driver support it.
- **Atomics:** Operations like `atomicAdd` give defined behavior when many threads update the same memory location; they trade simplicity for serialization cost in the hot path.

CUDA is widely used for deep learning (through frameworks that call CUDA/cuDNN), scientific computing, image and signal processing, and real-time graphics-adjacent compute.

---

## Repository chapters

| Chapter | Folder | Focus |
|--------|--------|--------|
| **1. Basics_CUDA** | `Basics_CUDA/` | Environment, first programs, compilation with `nvcc`, minimal host/device flow |
| **2. Kernels** | `Kernels/` | Kernel launch configuration, indexing patterns, reductions, matrix patterns |
| **3. Atomics** | `Atomics/` | Atomic operations, hazards, when atomics help vs hurt |
| **4. Streams** | `Streams/` | CUDA streams, overlap, ordering, and related timing patterns |

---

## CUDA Toolkit setup (WSL Ubuntu)

These steps assume **Windows Subsystem for Linux (WSL)** with an **Ubuntu** distribution and an **NVIDIA GPU** with a driver that supports the CUDA version you choose. The driver on Windows/WSL must be recent enough for your toolkit; always match **driver capability** to **CUDA toolkit** using NVIDIA’s compatibility tables in the [CUDA Toolkit release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) and the [download archive](https://developer.nvidia.com/cuda-12-6-0-download-archive) for the version you pick.

### Step 1: Check GPU and driver (compatible CUDA)

In your WSL Ubuntu terminal:

```bash
nvidia-smi
```

Inspect the output for:

- GPU model and **driver version**
- CUDA version shown as the maximum runtime the driver supports (use this to pick a toolkit that is **not newer** than what your driver advertises)

If `nvidia-smi` fails in WSL, install/update the **Windows** NVIDIA driver and ensure WSL GPU support is enabled per Microsoft’s WSL GPU documentation.

Update packages:

```bash
sudo apt update
```

### Step 2: Install the CUDA Toolkit

1. Open the NVIDIA CUDA download archive, for example: [CUDA Toolkit 12.6.0 download archive](https://developer.nvidia.com/cuda-12-6-0-download-archive).
2. Choose **Linux** → **x86_64** → **WSL-Ubuntu** (or **Ubuntu** with the version that matches your distro) and follow the **runfile** or **deb (network/local)** instructions shown on that page.
3. Pick a toolkit version **compatible with your GPU architecture** and **driver** (newer toolkits often require newer drivers).

Complete the installation commands exactly as NVIDIA lists for your selected installer type.

### Step 3: Add CUDA to `PATH` and library path

Edit your shell configuration (bash example):

```bash
nano ~/.bashrc
```

Add at the **bottom** of the file:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

If you installed CUDA somewhere other than `/usr/local/cuda`, point `PATH` and `LD_LIBRARY_PATH` to that prefix instead (some installers use a versioned path; a symlink `cuda` → `cuda-12.x` is common).

### Step 4: Apply configuration

```bash
source ~/.bashrc
```

### Step 5: Verify the compiler

```bash
nvcc --version
```

You should see the CUDA toolkit version and build details. Optionally confirm the runtime can see the device with a small sample or `deviceQuery` from the CUDA samples if you install them.

### Compiling a `.cu` file (typical)

From a folder containing a CUDA source file:

```bash
nvcc -o program program.cu
./program
```

Add `-arch=native` or a specific `-gencode` if you need to target a particular compute capability for distribution builds.

---

## Further reading

- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA Toolkit documentation](https://docs.nvidia.com/cuda/index.html)
