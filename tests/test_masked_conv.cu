#include <iostream>
#include <cooperative_groups.h>
#include <gtest/gtest.h>
#include "masked_conv.hpp"

TEST(Conv1D, TestNormalConvWorks) {
  int vector_size = 3;
  int kernel_size = 3;

  // TODO: use deviceQuery to get best block size
  int block_size = 3;
  // rounded up to nearest multiple of block size
  int num_blocks = 1;

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
    mask[i] = true;
  }
  for (int i = 0; i < kernel_size; i++) {
    kernel[i] = 4.0f + i;
  }

  // Launch kernel
  convolution_1d(num_blocks, block_size, v_out, v1, vector_size, kernel, kernel_size, mask, 0.0f);

  // Wait for kernel to finish
  cudaDeviceSynchronize();

  // Check results
  EXPECT_EQ(v_out[0], 13.0f);
  EXPECT_EQ(v_out[1], 28.0f);
  EXPECT_EQ(v_out[2], 27.0f);

  // Free memory
  cudaFree(v1);
  cudaFree(kernel);
  cudaFree(v_out);
  cudaFree(mask);
}