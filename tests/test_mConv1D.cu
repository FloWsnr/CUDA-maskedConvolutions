#include <iostream>
#include <cooperative_groups.h>
#include <gtest/gtest.h>
#include "mConv.hpp"

TEST(Conv1D, TestNormalConvWorksSmall) {
  const int vector_size = 3;
  const int kernel_size = 3;
  const int block_size = 256;

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
    v_out[i] = -1.0f;
    mask[i] = true;
  }
  for (int i = 0; i < kernel_size; i++) {
    kernel[i] = 4.0f + i;
  }

  // Launch kernel
  convolution_1d(block_size, v_out, v1, vector_size, kernel, kernel_size, mask, 0.0f);

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

TEST(Conv1D, TestNormalConvWorksLarge) {
  const int vector_size = 12;
  const int kernel_size = 3;
  const int block_size = 256;

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
    v_out[i] = -1.0f;
    mask[i] = true;
  }
  for (int i = 0; i < kernel_size; i++) {
    kernel[i] = 4.0f + i;
  }

  // Launch kernel
  convolution_1d(block_size, v_out, v1, vector_size, kernel, kernel_size, mask, 0.0f);

  // Wait for kernel to finish
  cudaDeviceSynchronize();

  // Check results
  EXPECT_EQ(v_out[0], 13.0f);
  EXPECT_EQ(v_out[1], 28.0f);
  EXPECT_EQ(v_out[2], 43.0f);
  EXPECT_EQ(v_out[3], 58.0f);
  EXPECT_EQ(v_out[4], 73.0f);
  EXPECT_EQ(v_out[5], 88.0f);
  EXPECT_EQ(v_out[6], 103.0f);
  EXPECT_EQ(v_out[7], 118.0f);
  EXPECT_EQ(v_out[8], 133.0f);
  EXPECT_EQ(v_out[9], 148.0f);
  EXPECT_EQ(v_out[10], 163.0f);
  EXPECT_EQ(v_out[11], 126.0f);


  // Free memory
  cudaFree(v1);
  cudaFree(kernel);
  cudaFree(v_out);
  cudaFree(mask);
}

TEST(Conv1D, TestMaskedConvWorks) {
  const int vector_size = 3;
  const int kernel_size = 3;
  const int block_size = 256;

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
    v_out[i] = -1.0f;
    mask[i] = true;
  }
  for (int i = 0; i < kernel_size; i++) {
    kernel[i] = 4.0f + i;
  }

  // Set first element of mask to false
  mask[0] = false;

  // Launch kernel
  convolution_1d(block_size, v_out, v1, vector_size, kernel, kernel_size, mask, 0.0f);

  // Wait for kernel to finish
  cudaDeviceSynchronize();

  // Check results
  EXPECT_EQ(v_out[0], 0.0f); // first element is skipped
  EXPECT_EQ(v_out[1], 22.0f);
  EXPECT_EQ(v_out[2], 27.0f);

  // Free memory
  cudaFree(v1);
  cudaFree(kernel);
  cudaFree(v_out);
  cudaFree(mask);
}

// TEST(Conv1D, TestSharedMemLoadsCorrectly) {

// }