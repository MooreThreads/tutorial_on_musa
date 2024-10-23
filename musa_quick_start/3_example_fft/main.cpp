/* main.cpp */
#include <iostream>
#include <mufft.h>
#include <musa_runtime.h>
int main() {
  const int Nx = 8;
  size_t complex_bytes = sizeof(float) * 2 * Nx;
  // create and initialize host data
  float *h_x = (float *)malloc(complex_bytes);
  for (size_t i = 0; i < Nx; i++) {
    h_x[2 * i] = i;
    h_x[2 * i + 1] = i;
  }
  // Create MUSA device object and copy data to device
  void *d_x;
  musaMalloc(&d_x, complex_bytes);
  musaMemcpy(d_x, h_x, complex_bytes, musaMemcpyHostToDevice);
  // Create the plan
  mufftHandle plan = NULL;
  mufftPlan1d(&plan, Nx, MUFFT_C2C, 1);
  // Execute plan:
  mufftExecC2C(plan, (mufftComplex *)d_x, (mufftComplex *)d_x, MUFFT_FORWARD);
  // copy back the result to host
  musaMemcpy(h_x, d_x, complex_bytes, musaMemcpyDeviceToHost);
  for (size_t i = 0; i < Nx; i++) {
    std::cout << "(" << h_x[2 * i] << ", " << h_x[2 * i + 1] << ")\n";
  }
  // release resource
  mufftDestroy(plan);
  musaFree(d_x);
  free(h_x);
  return 0;
}