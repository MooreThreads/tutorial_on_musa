#include <chrono>
#include <iostream>
#include <mublas.h>
#include <musa_runtime.h>
#include <vector>

size_t M = 16384;
size_t N = 16384;
size_t K = 16384;

struct PrecisionConfig
{
  int bytesPerElement;
  const char *name;
  int NUM_ITERATIONS;
  int WARMUP_ITERATIONS = 10;
};

void test(const PrecisionConfig &config)
{
  double *d_A, *d_B, *d_C;
  std::vector<double> h_A(M * K, double(0.9f));
  std::vector<double> h_B(K * N, double(1.2f));
  std::vector<double> h_C(M * N);

  musaMalloc(&d_A, M * K * config.bytesPerElement);
  musaMalloc(&d_B, K * N * config.bytesPerElement);
  musaMalloc(&d_C, M * N * config.bytesPerElement);

  musaMemcpy(d_A, h_A.data(), M * K * config.bytesPerElement, musaMemcpyHostToDevice);
  musaMemcpy(d_B, h_B.data(), K * N * config.bytesPerElement, musaMemcpyHostToDevice);

  mublasHandle_t handle;
  mublasCreate(&handle);

  double alpha = 1.0f;
  double beta = 0.0f;

  for (int i = 0; i < config.WARMUP_ITERATIONS; ++i)
  {
    mublasDgemm(handle, MUBLAS_OP_N, MUBLAS_OP_T,
                M, N, K, &alpha,
                d_A, M,
                d_B, N,
                &beta,
                d_C, M);
  }

  musaError_t syncError = musaDeviceSynchronize();
  auto start = std::chrono::high_resolution_clock::now();

  if (syncError != musaSuccess)
  {
    std::cout << "MUSA error: " << musaGetErrorString(syncError) << std::endl;
  }

  for (int i = 0; i < config.NUM_ITERATIONS; ++i)
  {
    mublasDgemm(handle, MUBLAS_OP_N, MUBLAS_OP_T,
                M, N, K, &alpha,
                d_A, M,
                d_B, N,
                &beta,
                d_C, M);
  }
  syncError = musaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  if (syncError != musaSuccess)
  {
    std::cout << "MUSA error: " << musaGetErrorString(syncError) << std::endl;
  }
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Average " << config.name << " Single Op Duration: "
            << duration.count() / config.NUM_ITERATIONS << " us" << std::endl;

  double time_second = duration.count() / 1.0e6;
  double flops = 2.0 * M * N * K * config.NUM_ITERATIONS;
  double FLOPS = flops / time_second;
  double TFLOPS = FLOPS / 1.0e12;

  std::cout << "[FlagPerf Result]" << "computation-FP64=" << TFLOPS << "TFLOPS"
            << std::endl;

  musaMemcpy(h_C.data(), d_C, M * N * config.bytesPerElement, musaMemcpyDeviceToHost);

  musaFree(d_A);
  musaFree(d_B);
  musaFree(d_C);

  mublasDestroy(handle);
}

int main(int argc, char* argv[]) {

  if (argc != 5) {
      std::cerr << "Usage: " << argv[0] << " <m> <n> <k> <iter>" << std::endl;
      std::cerr << "Example: " << argv[0] << " 128 128 128 10" << std::endl;
      return EXIT_FAILURE;
  }

  int m = std::atoi(argv[1]);
  int n = std::atoi(argv[2]);
  int k = std::atoi(argv[3]);
  int iter = std::atoi(argv[4]);

  std::cout << "========================================" << std::endl;
  std::cout << "MatMul FP64 Test (MUSA)" << std::endl;
  std::cout << "m = " << m << ", n = " << n << ", k = " << k << std::endl;
  std::cout << "Test Iterations = " << iter << std::endl;

  M = m;
  N = n;
  K = k;
  musaSetDevice(0);
  PrecisionConfig fp64_PrecisionConfig = {sizeof(double), "FP64", iter, 40};

  test(fp64_PrecisionConfig);

  return 0;
}
