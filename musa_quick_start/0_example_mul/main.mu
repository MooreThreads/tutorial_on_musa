#include <iostream>
 
__global__ void func_kernel(float *x, float *y, float a) {
  y[threadIdx.x] = a * x[threadIdx.x];
}
 
int main(int argc, char *argv[]) {
  const int kDataLen = 4;
 
  float a = 2.0f;
  float host_x[kDataLen] = {1.0f, 2.0f, 3.0f, 4.0f};
  float host_y[kDataLen];
 
  // Copy input data to device.
  float *device_x;
  float *device_y;
  musaMalloc(&device_x, kDataLen * sizeof(float));
  musaMalloc(&device_y, kDataLen * sizeof(float));
  musaMemcpy(device_x, host_x, kDataLen * sizeof(float),
             musaMemcpyHostToDevice);
 
  // Launch the kernel.
  func_kernel<<<1, kDataLen>>>(device_x, device_y, a);
 
  // Copy output data to host.
  musaDeviceSynchronize();
  musaMemcpy(host_y, device_y, kDataLen * sizeof(float),
             musaMemcpyDeviceToHost);
 
  // Print the results.
  std::cout << "func" << std::endl;
  for (int i = 0; i < kDataLen; ++i) {
    std::cout << "y[" << i << "] = " << host_y[i] << "\n";
  }
 
  musaFree(device_x);
  musaFree(device_y);
  return 0;
}