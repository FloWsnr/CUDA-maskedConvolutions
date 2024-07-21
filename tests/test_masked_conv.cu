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

TEST(Conv1D, TestMaskedConvWorks) {
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

  // Set first element of mask to false
  mask[0] = false;

  // Launch kernel
  convolution_1d(num_blocks, block_size, v_out, v1, vector_size, kernel, kernel_size, mask, 0.0f);

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

TEST(Conv3D, TestNormalConvWorks) {

  /* Test normal 3D convolution with a uniform kernel
  * Input and kernel are both filled with 1s as element
  * Thus, the middle element of the output should be 27
  */

  int nx_arr = 5;
  int ny_arr = 5;
  int nz_arr = 5;
  int nx_kernel = 3;
  int ny_kernel = 3;
  int nz_kernel = 3;
  float pad_val = 0.0f;

  // Total number of elements
  int n_arr = nx_arr * ny_arr * nz_arr;
  int n_mask = nx_arr * ny_arr * nz_arr;
  int n_kernel = nx_kernel * ny_kernel * nz_kernel;
  int n_out = nx_arr * ny_arr * nz_arr;

  int block_size = n_arr;
  int grid_size = 1;

  // Declare variables
  float* v1;
  float* kernel;
  float* v_out;
  bool* mask;

  // Allocate memory
  cudaMallocManaged(&v1, n_arr * sizeof(float));
  cudaMallocManaged(&mask, n_mask * sizeof(bool));
  cudaMallocManaged(&kernel, n_kernel * sizeof(float));
  cudaMallocManaged(&v_out, n_out * sizeof(float));

  for (int i = 0; i < n_arr; i++) {
    v1[i] = 1.0f;
    v_out[i] = 0.0f;
    mask[i] = true;
  }
  for (int i = 0; i < n_kernel; i++) {
    kernel[i] = 1.0f;
  }

  // Launch kernel
  convolution_3d(grid_size, block_size, v_out, v1, nx_arr, ny_arr, nz_arr, kernel, nx_kernel, ny_kernel, nz_kernel, mask, pad_val);

  // Wait for kernel to finish
  cudaDeviceSynchronize();

  // Check results
  EXPECT_EQ(v_out[13], 27.0f);
  EXPECT_EQ(v_out[0], 7.0f);

  // Free memory
  cudaFree(v1);
  cudaFree(kernel);
  cudaFree(v_out);
  cudaFree(mask);

}
