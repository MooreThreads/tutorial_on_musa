#include <cmath>
#include <iostream>
__global__ void axpy(float* x, float* y, float a) {
  y[threadIdx.x] = a * x[threadIdx.x];
}
int main(int argc, char* argv[]) {
  bool isSuccess = true;
  const int kDataLen = 4;
  float a = 2.0f;
  float host_x[kDataLen] = {1.0f, 2.0f, 3.0f, 4.0f};
  float host_y[kDataLen];
  // Copy input data to device.
  float* device_x;
  float* device_y;
  musaMalloc(&device_x, kDataLen * sizeof(float));
  musaMalloc(&device_y, kDataLen * sizeof(float));
  musaMemcpy(
      device_x, host_x, kDataLen * sizeof(float), musaMemcpyHostToDevice);
  // Launch the kernel.
  axpy<<<1, kDataLen>>>(device_x, device_y, a);
  // Copy output data to host.
  musaDeviceSynchronize();
  musaMemcpy(
      host_y, device_y, kDataLen * sizeof(float), musaMemcpyDeviceToHost);
  // Check the results.
  for (int i = 0; i < kDataLen; ++i) {
    if (std::fabs(host_y[i] - a * host_x[i]) > 1e-6) {
      isSuccess = false;
    }
  }
  musaFree(device_x);
  musaFree(device_y);
  musaDeviceReset();
  if (isSuccess == false) {
    std::cout << "Expected result is : " << a * host_x[0] << ", "
              << a * host_x[1] << ", " << a * host_x[2] << ", " << a * host_x[3]
              << std::endl;
    std::cout << "But the current calculation result is: " << host_y[0] << ", "
              << host_y[1] << ", " << host_y[2] << ", " << host_y[3] << std::endl;
    std::cout << "MUSA check failed, please check the Driver version on host and "
              << "MUSA version (command: musa_version_query)!" << std::endl;
    return 1;
  } else {
    std::cout << "MUSA check successfully!" << std::endl; 
    return 0;
  }
}

