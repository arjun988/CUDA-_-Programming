#pragma once

#include "common.cuh"

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
  __shared__ float As[BK * BM];
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
