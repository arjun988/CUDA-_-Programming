#pragma once

#include "common.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct Args {
  int M = 1024;
  int N = 1024;
  int K = 1024;
  std::string kernel = "all";
};

static inline Args parse_args(int argc, char** argv) {
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

static inline void fill_random(std::vector<float>& v) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& x : v) {
    x = dist(rng);
  }
}

static inline float max_abs_diff(const std::vector<float>& a,
                                 const std::vector<float>& b) {
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

static inline float gflops(double ms, int M, int N, int K) {
  const double flops = 2.0 * static_cast<double>(M) * N * K;
  return static_cast<float>(flops / (ms * 1.0e6));
}
