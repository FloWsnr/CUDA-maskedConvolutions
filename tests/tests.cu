#include <iostream>
#include <cooperative_groups.h>
#include "masked_conv.hpp"

int main() {
  int vector_size = 3;
  int kernel_size = 3;

  // TODO: use deviceQuery to get best block size
  int block_size = 256;
  // rounded up to nearest multiple of block size
  int num_blocks = (vector_size + block_size - 1) / block_size;

  // Declare variables
  float* v1;
  float* kernel;
  float* v_out;
  bool* mask;

  // Allocate memory
  cudaMallocManaged(&v1, vector_size * sizeof(float));
  cudaMallocManaged(&mask, vector_size * sizeof(bool));
  cudaMallocManaged(&kernel, kernel_size * sizeof(float));
  cudaMallocManaged(&v_out, vector_size * sizeof(float));

  for (int i = 0; i < vector_size; i++) {
    v1[i] = i + 1.0f;
    v_out[i] = 0.0f;
  }
  for (int i = 0; i < kernel_size; i++) {
    kernel[i] = 4.0f + i;
  }

  // Launch kernel
  convolution_1d(v_out, v1, vector_size, kernel, kernel_size, mask, 0.0f);

  // Wait for kernel to finish
  cudaDeviceSynchronize();

  // Print result
  for (int i = 0; i < vector_size; i++) {
    std::cout << v_out[i] << " ";
  }
  std::cout << std::endl;


  // Free memory
  cudaFree(v1);
  cudaFree(kernel);
  cudaFree(v_out);
  cudaFree(mask);

  return 0;
}