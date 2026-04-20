#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                        \
  do {                                                                          \
    cudaError_t err__ = (call);                                                 \
    if (err__ != cudaSuccess) {                                                 \
      std::cerr << "CUDA error: " << cudaGetErrorString(err__) << " at "        \
                << __FILE__ << ":" << __LINE__ << std::endl;                    \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

constexpr int THREADS = 128;
constexpr int TILE_N = 128;
constexpr int D_MAX = 128;

__global__ void flash_attention_forward_kernel(const float* Q, const float* K,
                                               const float* V, float* O, int B,
                                               int H, int N, int D, int causal) {
  __shared__ float Ks[TILE_N * D_MAX];
  __shared__ float Vs[TILE_N * D_MAX];
  __shared__ float reduce_buf[THREADS];

  const int i = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;

  if (i >= N || D > D_MAX) {
    return;
  }

  const int q_base = ((b * H + h) * N + i) * D;
  float q[D_MAX];
  for (int d = 0; d < D; ++d) {
    q[d] = Q[q_base + d];
  }

  float m = -INFINITY;
  float l = 0.0f;
  float acc[D_MAX];
  for (int d = 0; d < D; ++d) {
    acc[d] = 0.0f;
  }

  for (int tile = 0; tile < N; tile += TILE_N) {
    const int tile_size = min(TILE_N, N - tile);

    for (int idx = tid; idx < tile_size * D; idx += blockDim.x) {
      const int r = idx / D;
      const int d = idx % D;
      const int kv_idx = ((b * H + h) * N + (tile + r)) * D + d;
      Ks[r * D + d] = K[kv_idx];
      Vs[r * D + d] = V[kv_idx];
    }
    __syncthreads();

    float s = -INFINITY;
    int j = tid;
    if (j < tile_size) {
      const int global_j = tile + j;
      if (!causal || global_j <= i) {
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
          dot += q[d] * Ks[j * D + d];
        }
        s = dot * rsqrtf(static_cast<float>(D));
      }
    }

    reduce_buf[tid] = s;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + stride]);
      }
      __syncthreads();
    }
    const float tile_m = reduce_buf[0];
    const float new_m = fmaxf(m, tile_m);
    const float alpha = expf(m - new_m);

    float p = 0.0f;
    if (j < tile_size && s > -INFINITY / 2) {
      p = expf(s - new_m);
    }

    reduce_buf[tid] = p;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduce_buf[tid] += reduce_buf[tid + stride];
      }
      __syncthreads();
    }
    const float p_sum = reduce_buf[0];

    for (int d = 0; d < D; ++d) {
      float pv = 0.0f;
      if (j < tile_size && s > -INFINITY / 2) {
        pv = p * Vs[j * D + d];
      }
      reduce_buf[tid] = pv;
      __syncthreads();
      for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
          reduce_buf[tid] += reduce_buf[tid + stride];
        }
        __syncthreads();
      }
      if (tid == 0) {
        acc[d] = acc[d] * alpha + reduce_buf[0];
      }
      __syncthreads();
    }

    if (tid == 0) {
      l = l * alpha + p_sum;
      m = new_m;
    }
    __syncthreads();
  }

  if (tid == 0) {
    const int o_base = ((b * H + h) * N + i) * D;
    const float inv_l = 1.0f / fmaxf(l, 1e-20f);
    for (int d = 0; d < D; ++d) {
      O[o_base + d] = acc[d] * inv_l;
    }
  }
}

__global__ void naive_attention_forward_kernel(const float* Q, const float* K,
                                               const float* V, float* O, int B,
                                               int H, int N, int D, int causal) {
  const int i = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;

  if (i >= N || D > D_MAX || tid > 0) {
    return;
  }

  float q[D_MAX];
  const int q_base = ((b * H + h) * N + i) * D;
  for (int d = 0; d < D; ++d) {
    q[d] = Q[q_base + d];
  }

  float max_s = -INFINITY;
  for (int j = 0; j < N; ++j) {
    if (causal && j > i) {
      continue;
    }
    float dot = 0.0f;
    const int k_base = ((b * H + h) * N + j) * D;
    for (int d = 0; d < D; ++d) {
      dot += q[d] * K[k_base + d];
    }
    max_s = fmaxf(max_s, dot * rsqrtf(static_cast<float>(D)));
  }

  float denom = 0.0f;
  float acc[D_MAX];
  for (int d = 0; d < D; ++d) {
    acc[d] = 0.0f;
  }
  for (int j = 0; j < N; ++j) {
    if (causal && j > i) {
      continue;
    }
    float dot = 0.0f;
    const int kv_base = ((b * H + h) * N + j) * D;
    for (int d = 0; d < D; ++d) {
      dot += q[d] * K[kv_base + d];
    }
    const float p = expf(dot * rsqrtf(static_cast<float>(D)) - max_s);
    denom += p;
    for (int d = 0; d < D; ++d) {
      acc[d] += p * V[kv_base + d];
    }
  }
  const float inv_denom = 1.0f / fmaxf(denom, 1e-20f);
  const int o_base = ((b * H + h) * N + i) * D;
  for (int d = 0; d < D; ++d) {
    O[o_base + d] = acc[d] * inv_denom;
  }
}

void attention_cpu_reference(const std::vector<float>& Q,
                             const std::vector<float>& K,
                             const std::vector<float>& V, std::vector<float>& O,
                             int B, int H, int N, int D, bool causal) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      for (int i = 0; i < N; ++i) {
        float max_s = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < N; ++j) {
          if (causal && j > i) {
            continue;
          }
          float dot = 0.0f;
          for (int d = 0; d < D; ++d) {
            const int q_idx = ((b * H + h) * N + i) * D + d;
            const int k_idx = ((b * H + h) * N + j) * D + d;
            dot += Q[q_idx] * K[k_idx];
          }
          max_s = std::max(max_s, dot * scale);
        }

        float denom = 0.0f;
        for (int d = 0; d < D; ++d) {
          O[((b * H + h) * N + i) * D + d] = 0.0f;
        }
        for (int j = 0; j < N; ++j) {
          if (causal && j > i) {
            continue;
          }
          float dot = 0.0f;
          for (int d = 0; d < D; ++d) {
            const int q_idx = ((b * H + h) * N + i) * D + d;
            const int k_idx = ((b * H + h) * N + j) * D + d;
            dot += Q[q_idx] * K[k_idx];
          }
          const float p = std::exp(dot * scale - max_s);
          denom += p;
          for (int d = 0; d < D; ++d) {
            const int v_idx = ((b * H + h) * N + j) * D + d;
            O[((b * H + h) * N + i) * D + d] += p * V[v_idx];
          }
        }
        const float inv_denom = 1.0f / std::max(denom, 1e-20f);
        for (int d = 0; d < D; ++d) {
          O[((b * H + h) * N + i) * D + d] *= inv_denom;
        }
      }
    }
  }
}

template <typename F>
float benchmark_ms(F&& fn, int warmup = 5, int iters = 20) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    fn();
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return ms / static_cast<float>(iters);
}

struct Args {
  int B = 1;
  int H = 8;
  int N = 512;
  int D = 64;
  int causal = 1;
};

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--b") == 0 && i + 1 < argc) a.B = std::atoi(argv[++i]);
    if (std::strcmp(argv[i], "--h") == 0 && i + 1 < argc) a.H = std::atoi(argv[++i]);
    if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) a.N = std::atoi(argv[++i]);
    if (std::strcmp(argv[i], "--d") == 0 && i + 1 < argc) a.D = std::atoi(argv[++i]);
    if (std::strcmp(argv[i], "--causal") == 0 && i + 1 < argc) a.causal = std::atoi(argv[++i]);
  }
  return a;
}

int main(int argc, char** argv) {
  const Args args = parse_args(argc, argv);
  if (args.D > D_MAX) {
    std::cerr << "D must be <= " << D_MAX << " for this implementation.\n";
    return 1;
  }

  const size_t elems = static_cast<size_t>(args.B) * args.H * args.N * args.D;
  std::vector<float> hQ(elems), hK(elems), hV(elems);
  std::vector<float> hOutFlash(elems), hOutNaive(elems), hOutCpu(elems);

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < elems; ++i) {
    hQ[i] = dist(rng);
    hK[i] = dist(rng);
    hV[i] = dist(rng);
  }

  float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dOFlash = nullptr, *dONaive = nullptr;
  CHECK_CUDA(cudaMalloc(&dQ, elems * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dK, elems * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dV, elems * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dOFlash, elems * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dONaive, elems * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dQ, hQ.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dK, hK.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dV, hV.data(), elems * sizeof(float), cudaMemcpyHostToDevice));

  const dim3 grid(args.N, args.H, args.B);
  const dim3 block_flash(THREADS);
  const dim3 block_naive(1);

  auto run_flash = [&]() {
    flash_attention_forward_kernel<<<grid, block_flash>>>(dQ, dK, dV, dOFlash, args.B,
                                                          args.H, args.N, args.D,
                                                          args.causal);
  };
  auto run_naive = [&]() {
    naive_attention_forward_kernel<<<grid, block_naive>>>(dQ, dK, dV, dONaive, args.B,
                                                          args.H, args.N, args.D,
                                                          args.causal);
  };

  run_flash();
  run_naive();
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(hOutFlash.data(), dOFlash, elems * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(hOutNaive.data(), dONaive, elems * sizeof(float), cudaMemcpyDeviceToHost));
  attention_cpu_reference(hQ, hK, hV, hOutCpu, args.B, args.H, args.N, args.D, args.causal != 0);

  auto max_diff = [](const std::vector<float>& a, const std::vector<float>& b) {
    float md = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
      md = std::max(md, std::fabs(a[i] - b[i]));
    }
    return md;
  };

  const float diff_flash_naive = max_diff(hOutFlash, hOutNaive);
  const float diff_flash_cpu = max_diff(hOutFlash, hOutCpu);

  const float flash_ms = benchmark_ms(run_flash);
  const float naive_ms = benchmark_ms(run_naive);

  std::cout << "FlashAttention (paper-inspired forward) demo\n";
  std::cout << "Shape: B=" << args.B << " H=" << args.H << " N=" << args.N
            << " D=" << args.D << " causal=" << args.causal << "\n";
  std::cout << "Correctness:\n";
  std::cout << "  max|flash - naive_gpu| = " << diff_flash_naive << "\n";
  std::cout << "  max|flash - cpu_ref|   = " << diff_flash_cpu << "\n";
  std::cout << "Timing (ms):\n";
  std::cout << "  flash kernel = " << flash_ms << "\n";
  std::cout << "  naive kernel = " << naive_ms << "\n";
  if (flash_ms > 0.0f) {
    std::cout << "  speedup      = " << (naive_ms / flash_ms) << "x\n";
  }

  CHECK_CUDA(cudaFree(dQ));
  CHECK_CUDA(cudaFree(dK));
  CHECK_CUDA(cudaFree(dV));
  CHECK_CUDA(cudaFree(dOFlash));
  CHECK_CUDA(cudaFree(dONaive));
  return 0;
}
