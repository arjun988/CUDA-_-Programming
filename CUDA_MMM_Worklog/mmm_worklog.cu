#include "common.cuh"
#include "kernels.cuh"
#include "utils.hpp"

#include <iostream>
#include <limits>
#include <string>
#include <vector>

static void run_cublas(cublasHandle_t handle, int M, int N, int K, float alpha,
                       const float* dA, const float* dB, float beta, float* dC) {
  CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB,
                           N, dA, K, &beta, dC, N));
}

int main(int argc, char** argv) {
  const Args args = parse_args(argc, argv);
  const int M = args.M;
  const int N = args.N;
  const int K = args.K;
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;

  std::vector<float> hA(M * K), hB(K * N), hC(M * N), hRef(M * N);
  fill_random(hA);
  fill_random(hB);
  std::fill(hC.begin(), hC.end(), 0.0f);
  std::fill(hRef.begin(), hRef.end(), 0.0f);

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CHECK_CUDA(cudaMalloc(&dA, sizeof(float) * hA.size()));
  CHECK_CUDA(cudaMalloc(&dB, sizeof(float) * hB.size()));
  CHECK_CUDA(cudaMalloc(&dC, sizeof(float) * hC.size()));
  CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeof(float) * hA.size(),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeof(float) * hB.size(),
                        cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  CHECK_CUDA(cudaMemset(dC, 0, sizeof(float) * hC.size()));
  const float cublasMs = time_kernel_ms(
      [&]() { run_cublas(handle, M, N, K, alpha, dA, dB, beta, dC); }, 3, 10);
  CHECK_CUDA(cudaMemcpy(hRef.data(), dC, sizeof(float) * hRef.size(),
                        cudaMemcpyDeviceToHost));

  std::cout << "Matrix size: M=" << M << " N=" << N << " K=" << K << "\n";
  std::cout << "cuBLAS : " << cublasMs << " ms, " << gflops(cublasMs, M, N, K)
            << " GFLOP/s\n";

  auto run_and_report = [&](const std::string& name, auto launch) {
    CHECK_CUDA(cudaMemset(dC, 0, sizeof(float) * hC.size()));
    const float ms = time_kernel_ms(launch);
    CHECK_CUDA(cudaMemcpy(hC.data(), dC, sizeof(float) * hC.size(),
                          cudaMemcpyDeviceToHost));
    const float diff = max_abs_diff(hC, hRef);
    std::cout << name << " : " << ms << " ms, " << gflops(ms, M, N, K)
              << " GFLOP/s, max|diff|=" << diff << "\n";
  };

  const dim3 grid32(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  const dim3 block2d(32, 32);
  const dim3 block1d(32 * 32);

  if (args.kernel == "k1" || args.kernel == "all") {
    run_and_report("k1_naive", [&]() {
      sgemm_k1_naive<<<grid32, block2d>>>(M, N, K, alpha, dA, dB, beta, dC);
    });
  }

  if (args.kernel == "k2" || args.kernel == "all") {
    run_and_report("k2_coalesced", [&]() {
      sgemm_k2_coalesced<<<grid32, block1d>>>(M, N, K, alpha, dA, dB, beta, dC);
    });
  }

  if (args.kernel == "k3" || args.kernel == "all") {
    run_and_report("k3_smem", [&]() {
      sgemm_k3_smem<<<grid32, block1d>>>(M, N, K, alpha, dA, dB, beta, dC);
    });
  }

  if (args.kernel == "k4" || args.kernel == "all") {
    constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
    const dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    const dim3 block((BM * BN) / TM);
    run_and_report("k4_1d_tiling", [&]() {
      sgemm_k4_1d_tiling<BM, BN, BK, TM><<<grid, block>>>(M, N, K, alpha, dA,
                                                          dB, beta, dC);
    });
  }

  if (args.kernel == "k5" || args.kernel == "all") {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    const dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    const dim3 block((BM * BN) / (TM * TN));
    run_and_report("k5_2d_tiling", [&]() {
      sgemm_k5_2d_tiling<BM, BN, BK, TM, TN><<<grid, block>>>(
          M, N, K, alpha, dA, dB, beta, dC);
    });
  }

  if (args.kernel == "k6" || args.kernel == "all") {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    const dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    const dim3 block((BM * BN) / (TM * TN));
    run_and_report("k6_vectorized", [&]() {
      sgemm_k6_vectorized<BM, BN, BK, TM, TN><<<grid, block>>>(
          M, N, K, alpha, dA, dB, beta, dC);
    });
  }

  if (args.kernel == "autotune") {
    struct Config {
      int BK;
      int TM;
      int TN;
      std::string name;
    };
    const std::vector<Config> configs = {
        {8, 8, 8, "BK8_TM8_TN8"},
        {16, 8, 8, "BK16_TM8_TN8"},
        {8, 4, 8, "BK8_TM4_TN8"},
        {8, 8, 4, "BK8_TM8_TN4"},
    };
    std::cout << "Autotune over kernel-6 style configs\n";
    for (const auto& cfg : configs) {
      CHECK_CUDA(cudaMemset(dC, 0, sizeof(float) * hC.size()));
      float ms = std::numeric_limits<float>::infinity();

      if (cfg.BK == 8 && cfg.TM == 8 && cfg.TN == 8) {
        constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
        const dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
        const dim3 block((BM * BN) / (TM * TN));
        ms = time_kernel_ms([&]() {
          sgemm_k6_vectorized<BM, BN, BK, TM, TN><<<grid, block>>>(
              M, N, K, alpha, dA, dB, beta, dC);
        });
      } else if (cfg.BK == 16 && cfg.TM == 8 && cfg.TN == 8) {
        constexpr int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8;
        const dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
        const dim3 block((BM * BN) / (TM * TN));
        ms = time_kernel_ms([&]() {
          sgemm_k6_vectorized<BM, BN, BK, TM, TN><<<grid, block>>>(
              M, N, K, alpha, dA, dB, beta, dC);
        });
      } else if (cfg.BK == 8 && cfg.TM == 4 && cfg.TN == 8) {
        constexpr int BM = 128, BN = 128, BK = 8, TM = 4, TN = 8;
        const dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
        const dim3 block((BM * BN) / (TM * TN));
        ms = time_kernel_ms([&]() {
          sgemm_k6_vectorized<BM, BN, BK, TM, TN><<<grid, block>>>(
              M, N, K, alpha, dA, dB, beta, dC);
        });
      } else if (cfg.BK == 8 && cfg.TM == 8 && cfg.TN == 4) {
        constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 4;
        const dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
        const dim3 block((BM * BN) / (TM * TN));
        ms = time_kernel_ms([&]() {
          sgemm_k6_vectorized<BM, BN, BK, TM, TN><<<grid, block>>>(
              M, N, K, alpha, dA, dB, beta, dC);
        });
      }

      CHECK_CUDA(cudaMemcpy(hC.data(), dC, sizeof(float) * hC.size(),
                            cudaMemcpyDeviceToHost));
      const float diff = max_abs_diff(hC, hRef);
      std::cout << cfg.name << " : " << ms << " ms, " << gflops(ms, M, N, K)
                << " GFLOP/s, max|diff|=" << diff << "\n";
    }
  }

  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK_CUDA(cudaFree(dA));
  CHECK_CUDA(cudaFree(dB));
  CHECK_CUDA(cudaFree(dC));
  return 0;
}
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <tuple>
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

#define CHECK_CUBLAS(call)                                                      \
  do {                                                                          \
    cublasStatus_t st__ = (call);                                               \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                        \
      std::cerr << "cuBLAS error: " << static_cast<int>(st__) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                    \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

constexpr int CEIL_DIV(int a, int b) { return (a + b - 1) / b; }

__global__ void sgemm_k1_naive(int M, int N, int K, float alpha, const float* A,
                               const float* B, float beta, float* C) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N) {
    float acc = 0.0f;
    for (int i = 0; i < K; ++i) {
      acc += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

__global__ void sgemm_k2_coalesced(int M, int N, int K, float alpha,
                                   const float* A, const float* B, float beta,
                                   float* C) {
  constexpr int BLOCK = 32;
  const int row = blockIdx.x * BLOCK + (threadIdx.x / BLOCK);
  const int col = blockIdx.y * BLOCK + (threadIdx.x % BLOCK);

  if (row < M && col < N) {
    float acc = 0.0f;
    for (int i = 0; i < K; ++i) {
      acc += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

__global__ void sgemm_k3_smem(int M, int N, int K, float alpha, const float* A,
                              const float* B, float beta, float* C) {
  constexpr int BLOCK = 32;
  __shared__ float As[BLOCK * BLOCK];
  __shared__ float Bs[BLOCK * BLOCK];

  const int cRow = blockIdx.x;
  const int cCol = blockIdx.y;
  const int threadRow = threadIdx.x / BLOCK;
  const int threadCol = threadIdx.x % BLOCK;

  float acc = 0.0f;

  const int rowBase = cRow * BLOCK;
  const int colBase = cCol * BLOCK;

  for (int bk = 0; bk < K; bk += BLOCK) {
    const int aRow = rowBase + threadRow;
    const int aCol = bk + threadCol;
    const int bRow = bk + threadRow;
    const int bCol = colBase + threadCol;

    As[threadRow * BLOCK + threadCol] =
        (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;
    Bs[threadRow * BLOCK + threadCol] =
        (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;
    __syncthreads();

    for (int dot = 0; dot < BLOCK; ++dot) {
      acc += As[threadRow * BLOCK + dot] * Bs[dot * BLOCK + threadCol];
    }
    __syncthreads();
  }

  const int outRow = rowBase + threadRow;
  const int outCol = colBase + threadCol;
  if (outRow < M && outCol < N) {
    C[outRow * N + outCol] = alpha * acc + beta * C[outRow * N + outCol];
  }
}

template <int BM, int BN, int BK, int TM>
__global__ void sgemm_k4_1d_tiling(int M, int N, int K, float alpha,
                                   const float* A, const float* B, float beta,
                                   float* C) {
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  constexpr int THREADS = (BM * BN) / TM;
  static_assert(THREADS <= 1024, "Too many threads per block");

  const int tid = threadIdx.x;
  const int blockRow = blockIdx.y;
  const int blockCol = blockIdx.x;

  const int threadCol = tid % BN;
  const int threadRow = tid / BN;

  const int innerRowA = tid / BK;
  const int innerColA = tid % BK;
  const int innerRowB = tid / BN;
  const int innerColB = tid % BN;

  float threadResults[TM];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    threadResults[i] = 0.0f;
  }

  for (int bk = 0; bk < K; bk += BK) {
    const int aRow = blockRow * BM + innerRowA;
    const int aCol = bk + innerColA;
    const int bRow = bk + innerRowB;
    const int bCol = blockCol * BN + innerColB;

    As[innerRowA * BK + innerColA] =
        (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;
    Bs[innerRowB * BN + innerColB] =
        (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;
    __syncthreads();

    for (int dot = 0; dot < BK; ++dot) {
      const float bVal = Bs[dot * BN + threadCol];
#pragma unroll
      for (int r = 0; r < TM; ++r) {
        const int localRow = threadRow * TM + r;
        threadResults[r] += As[localRow * BK + dot] * bVal;
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < TM; ++r) {
    const int outRow = blockRow * BM + threadRow * TM + r;
    const int outCol = blockCol * BN + threadCol;
    if (outRow < M && outCol < N) {
      C[outRow * N + outCol] =
          alpha * threadResults[r] + beta * C[outRow * N + outCol];
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_k5_2d_tiling(int M, int N, int K, float alpha,
                                   const float* A, const float* B, float beta,
                                   float* C) {
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  constexpr int THREADS = (BM * BN) / (TM * TN);
  static_assert(THREADS <= 1024, "Too many threads per block");

  const int tid = threadIdx.x;
  const int blockRow = blockIdx.y;
  const int blockCol = blockIdx.x;
  const int threadRow = tid / (BN / TN);
  const int threadCol = tid % (BN / TN);

  float threadResults[TM * TN];
  float regM[TM];
  float regN[TN];

#pragma unroll
  for (int i = 0; i < TM * TN; ++i) {
    threadResults[i] = 0.0f;
  }

  constexpr int THREAD_COUNT = THREADS;

  for (int bk = 0; bk < K; bk += BK) {
    for (int idx = tid; idx < BM * BK; idx += THREAD_COUNT) {
      const int r = idx / BK;
      const int c = idx % BK;
      const int aRow = blockRow * BM + r;
      const int aCol = bk + c;
      As[idx] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;
    }
    for (int idx = tid; idx < BK * BN; idx += THREAD_COUNT) {
      const int r = idx / BN;
      const int c = idx % BN;
      const int bRow = bk + r;
      const int bCol = blockCol * BN + c;
      Bs[idx] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;
    }
    __syncthreads();

    for (int dot = 0; dot < BK; ++dot) {
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dot];
      }
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        regN[j] = Bs[dot * BN + threadCol * TN + j];
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          threadResults[i * TN + j] += regM[i] * regN[j];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int outRow = blockRow * BM + threadRow * TM + i;
      const int outCol = blockCol * BN + threadCol * TN + j;
      if (outRow < M && outCol < N) {
        C[outRow * N + outCol] =
            alpha * threadResults[i * TN + j] + beta * C[outRow * N + outCol];
      }
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_k6_vectorized(int M, int N, int K, float alpha,
                                    const float* A, const float* B, float beta,
                                    float* C) {
  __shared__ float As[BK * BM];  // transposed layout: [BK, BM]
  __shared__ float Bs[BK * BN];  // normal layout: [BK, BN]

  constexpr int THREADS = (BM * BN) / (TM * TN);
  static_assert(THREADS <= 1024, "Too many threads per block");

  const int tid = threadIdx.x;
  const int blockRow = blockIdx.y;
  const int blockCol = blockIdx.x;
  const int threadRow = tid / (BN / TN);
  const int threadCol = tid % (BN / TN);

  float threadResults[TM * TN];
  float regM[TM];
  float regN[TN];

#pragma unroll
  for (int i = 0; i < TM * TN; ++i) {
    threadResults[i] = 0.0f;
  }

  for (int bk = 0; bk < K; bk += BK) {
    for (int idx = tid; idx < (BM * BK) / 4; idx += THREADS) {
      const int r = idx / (BK / 4);
      const int c4 = idx % (BK / 4);
      const int aRow = blockRow * BM + r;
      const int aCol = bk + c4 * 4;
      const int aLinear = aRow * K + aCol;

      if (aRow < M && aCol + 3 < K && ((aLinear & 3) == 0)) {
        const float4 v = reinterpret_cast<const float4*>(&A[aLinear])[0];
        As[(c4 * 4 + 0) * BM + r] = v.x;
        As[(c4 * 4 + 1) * BM + r] = v.y;
        As[(c4 * 4 + 2) * BM + r] = v.z;
        As[(c4 * 4 + 3) * BM + r] = v.w;
      } else {
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          const int col = aCol + t;
          As[col * BM + r] = (aRow < M && col < K) ? A[aRow * K + col] : 0.0f;
        }
      }
    }

    for (int idx = tid; idx < (BK * BN) / 4; idx += THREADS) {
      const int r = idx / (BN / 4);
      const int c4 = idx % (BN / 4);
      const int bRow = bk + r;
      const int bCol = blockCol * BN + c4 * 4;
      const int bLinear = bRow * N + bCol;

      if (bRow < K && bCol + 3 < N && ((bLinear & 3) == 0)) {
        reinterpret_cast<float4*>(&Bs[r * BN + c4 * 4])[0] =
            reinterpret_cast<const float4*>(&B[bLinear])[0];
      } else {
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          const int col = bCol + t;
          Bs[r * BN + c4 * 4 + t] =
              (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        }
      }
    }
    __syncthreads();

    for (int dot = 0; dot < BK; ++dot) {
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        regM[i] = As[dot * BM + threadRow * TM + i];
      }
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        regN[j] = Bs[dot * BN + threadCol * TN + j];
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          threadResults[i * TN + j] += regM[i] * regN[j];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int outRow = blockRow * BM + threadRow * TM + i;
      const int outCol = blockCol * BN + threadCol * TN + j;
      if (outRow < M && outCol < N) {
        C[outRow * N + outCol] =
            alpha * threadResults[i * TN + j] + beta * C[outRow * N + outCol];
      }
    }
  }
}

static void fill_random(std::vector<float>& v) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& x : v) {
    x = dist(rng);
  }
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    m = std::max(m, std::fabs(a[i] - b[i]));
  }
  return m;
}

template <typename LaunchFn>
float time_kernel_ms(LaunchFn&& launch, int warmup = 5, int iters = 20) {
  for (int i = 0; i < warmup; ++i) {
    launch();
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    launch();
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return ms / static_cast<float>(iters);
}

static float gflops(double ms, int M, int N, int K) {
  const double flops = 2.0 * static_cast<double>(M) * N * K;
  return static_cast<float>(flops / (ms * 1.0e6));
}

struct Args {
  int M = 1024;
  int N = 1024;
  int K = 1024;
  std::string kernel = "all";
};

static Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
      args.M = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
      args.N = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
      args.K = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) {
      args.kernel = argv[++i];
    }
  }
  return args;
}

static void run_cublas(cublasHandle_t handle, int M, int N, int K, float alpha,
                       const float* dA, const float* dB, float beta, float* dC) {
  CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB,
                           N, dA, K, &beta, dC, N));
}

int main(int argc, char** argv) {
  const Args args = parse_args(argc, argv);
  const int M = args.M;
  const int N = args.N;
  const int K = args.K;
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;

  std::vector<float> hA(M * K), hB(K * N), hC(M * N), hRef(M * N);
  fill_random(hA);
  fill_random(hB);
  std::fill(hC.begin(), hC.end(), 0.0f);
  std::fill(hRef.begin(), hRef.end(), 0.0f);

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CHECK_CUDA(cudaMalloc(&dA, sizeof(float) * hA.size()));
  CHECK_CUDA(cudaMalloc(&dB, sizeof(float) * hB.size()));
  CHECK_CUDA(cudaMalloc(&dC, sizeof(float) * hC.size()));
  CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeof(float) * hA.size(),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeof(float) * hB.size(),
                        cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  CHECK_CUDA(cudaMemset(dC, 0, sizeof(float) * hC.size()));
  const float cublasMs = time_kernel_ms(
      [&]() { run_cublas(handle, M, N, K, alpha, dA, dB, beta, dC); }, 3, 10);
  CHECK_CUDA(cudaMemcpy(hRef.data(), dC, sizeof(float) * hRef.size(),
                        cudaMemcpyDeviceToHost));

  std::cout << "Matrix size: M=" << M << " N=" << N << " K=" << K << "\n";
  std::cout << "cuBLAS : " << cublasMs << " ms, " << gflops(cublasMs, M, N, K)
            << " GFLOP/s\n";

  auto run_and_report = [&](const std::string& name, auto launch) {
    CHECK_CUDA(cudaMemset(dC, 0, sizeof(float) * hC.size()));
    const float ms = time_kernel_ms(launch);
    CHECK_CUDA(cudaMemcpy(hC.data(), dC, sizeof(float) * hC.size(),
                          cudaMemcpyDeviceToHost));
    const float diff = max_abs_diff(hC, hRef);
    std::cout << name << " : " << ms << " ms, " << gflops(ms, M, N, K)
              << " GFLOP/s, max|diff|=" << diff << "\n";
  };

  const dim3 grid32(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  const dim3 block2d(32, 32);
  const dim3 block1d(32 * 32);

  if (args.kernel == "k1" || args.kernel == "all") {
    run_and_report("k1_naive", [&]() {
      sgemm_k1_naive<<<grid32, block2d>>>(M, N, K, alpha, dA, dB, beta, dC);
    });
  }

  if (args.kernel == "k2" || args.kernel == "all") {
    run_and_report("k2_coalesced", [&]() {
      sgemm_k2_coalesced<<<grid32, block1d>>>(M, N, K, alpha, dA, dB, beta, dC);
    });
  }

  if (args.kernel == "k3" || args.kernel == "all") {
    run_and_report("k3_smem", [&]() {
      sgemm_k3_smem<<<grid32, block1d>>>(M, N, K, alpha, dA, dB, beta, dC);
    });
  }

  if (args.kernel == "k4" || args.kernel == "all") {
    constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
    const dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    const dim3 block((BM * BN) / TM);
    run_and_report("k4_1d_tiling", [&]() {
      sgemm_k4_1d_tiling<BM, BN, BK, TM><<<grid, block>>>(M, N, K, alpha, dA,
                                                          dB, beta, dC);
    });
  }

  if (args.kernel == "k5" || args.kernel == "all") {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    const dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    const dim3 block((BM * BN) / (TM * TN));
    run_and_report("k5_2d_tiling", [&]() {
      sgemm_k5_2d_tiling<BM, BN, BK, TM, TN><<<grid, block>>>(
          M, N, K, alpha, dA, dB, beta, dC);
    });
  }

  if (args.kernel == "k6" || args.kernel == "all") {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    const dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    const dim3 block((BM * BN) / (TM * TN));
    run_and_report("k6_vectorized", [&]() {
      sgemm_k6_vectorized<BM, BN, BK, TM, TN><<<grid, block>>>(
          M, N, K, alpha, dA, dB, beta, dC);
    });
  }

  if (args.kernel == "autotune") {
    struct Config {
      int BK;
      int TM;
      int TN;
      std::string name;
    };
    const std::vector<Config> configs = {
        {8, 8, 8, "BK8_TM8_TN8"},
        {16, 8, 8, "BK16_TM8_TN8"},
        {8, 4, 8, "BK8_TM4_TN8"},
        {8, 8, 4, "BK8_TM8_TN4"},
    };
    std::cout << "Autotune over kernel-6 style configs\n";
    for (const auto& cfg : configs) {
      CHECK_CUDA(cudaMemset(dC, 0, sizeof(float) * hC.size()));
      float ms = std::numeric_limits<float>::infinity();

      if (cfg.BK == 8 && cfg.TM == 8 && cfg.TN == 8) {
        constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
        const dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
        const dim3 block((BM * BN) / (TM * TN));
        ms = time_kernel_ms([&]() {
          sgemm_k6_vectorized<BM, BN, BK, TM, TN><<<grid, block>>>(
              M, N, K, alpha, dA, dB, beta, dC);
        });
      } else if (cfg.BK == 16 && cfg.TM == 8 && cfg.TN == 8) {
        constexpr int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8;
        const dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
        const dim3 block((BM * BN) / (TM * TN));
        ms = time_kernel_ms([&]() {
          sgemm_k6_vectorized<BM, BN, BK, TM, TN><<<grid, block>>>(
              M, N, K, alpha, dA, dB, beta, dC);
        });
      } else if (cfg.BK == 8 && cfg.TM == 4 && cfg.TN == 8) {
        constexpr int BM = 128, BN = 128, BK = 8, TM = 4, TN = 8;
        const dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
        const dim3 block((BM * BN) / (TM * TN));
        ms = time_kernel_ms([&]() {
          sgemm_k6_vectorized<BM, BN, BK, TM, TN><<<grid, block>>>(
              M, N, K, alpha, dA, dB, beta, dC);
        });
      } else if (cfg.BK == 8 && cfg.TM == 8 && cfg.TN == 4) {
        constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 4;
        const dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
        const dim3 block((BM * BN) / (TM * TN));
        ms = time_kernel_ms([&]() {
          sgemm_k6_vectorized<BM, BN, BK, TM, TN><<<grid, block>>>(
              M, N, K, alpha, dA, dB, beta, dC);
        });
      }

      CHECK_CUDA(cudaMemcpy(hC.data(), dC, sizeof(float) * hC.size(),
                            cudaMemcpyDeviceToHost));
      const float diff = max_abs_diff(hC, hRef);
      std::cout << cfg.name << " : " << ms << " ms, " << gflops(ms, M, N, K)
                << " GFLOP/s, max|diff|=" << diff << "\n";
    }
  }

  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK_CUDA(cudaFree(dA));
  CHECK_CUDA(cudaFree(dB));
  CHECK_CUDA(cudaFree(dC));
  return 0;
}
