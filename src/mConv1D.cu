/*******************************************************************************
* Kernel to compute the masked 3D convolution of a tensor and a kernel.
* The convolution is done with padding and the output is 'same', i.e. the
* output is the same size as the input.
*
*/
#include <iostream>
#include <cooperative_groups.h>
#include "mConv.hpp"

namespace cg = cooperative_groups;

__device__ void load_to_shared(
    float* sdata,
    const float* arr,
    const int n_arr,
    const int k_radius) {

    const unsigned int t_idx{ threadIdx.x }; // thread index
    const unsigned int b_dim{ blockDim.x }; // number of threads per block
    const unsigned int b_idx{ blockIdx.x }; // block index

    const unsigned int idx{ b_idx * b_dim + t_idx };

    for (int i = idx; i < k_radius + blockDim.x; i += blockDim.x) {
        sdata[i] = arr[i - k_radius];
    }
    __syncthreads();
}

__global__ void convolution_1d_kernel(
    float* out,
    const float* arr,
    const int n_arr,
    const float* kernel,
    const int n_kernel,
    const bool* mask,
    const float pad_val) {


    const unsigned int t_idx{ threadIdx.x }; // thread index
    const unsigned int b_dim{ blockDim.x }; // number of threads per block
    const unsigned int b_idx{ blockIdx.x }; // block index

    const unsigned int idx{ b_idx * b_dim + t_idx };
    const unsigned int stride{ b_dim * gridDim.x }; // total number of threads

    const unsigned int k_radius = n_kernel / 2;

    // shared memory for the kernel (size = k_radius + b_dim)
    extern __shared__ float sdata[];
    load_to_shared(sdata, arr, n_arr, k_radius);

    for (auto i = idx; i < n_arr; i += stride) {
        if (!mask[i]) continue; // skip if mask is false

        // loop over kernel
        for (int j = 0; j < n_kernel; ++j) {

            // index the array with implicit reversed kernel
            int input_index = i - j + k_radius;

            if (input_index >= 0 && input_index < n_arr) {
                if (!mask[input_index]) continue; // skip if mask is false
                out[i] += arr[input_index] * kernel[j];
            }
            else {
                out[i] += pad_val * kernel[j];
            }
        }
    }
}

void convolution_1d(
    int grid_size,
    int block_size,
    float* out,
    float* arr,
    int n_arr,
    float* kernel,
    int n_kernel,
    bool* mask,
    float pad_val) {
    convolution_1d_kernel << <grid_size, block_size, block_size + n_kernel >> > (
        out, arr, n_arr, kernel, n_kernel, mask, pad_val
        );
}

void convolution_1d_cpu(float* out, float* arr, int n_arr, float* kernel, int n_kernel, bool* mask, float pad_val) {

    for (int i = 0; i < n_arr; i++) {
        if (!mask[i]) continue; // skip if mask is false

        // loop over kernel
        for (int j = 0; j < n_kernel; ++j) {

            // index the array with implicit reversed kernel
            int input_index = i - j + n_kernel / 2;
            if (input_index >= 0 && input_index < n_arr) {
                if (!mask[input_index]) continue; // skip if mask is false
                out[i] += arr[input_index] * kernel[j];
            }
            else {
                out[i] += pad_val * kernel[j];
            }
        }
    }
}
