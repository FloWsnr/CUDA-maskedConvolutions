/*******************************************************************************
* Kernel to compute the masked 3D convolution of a tensor and a kernel.
* The convolution is done with padding and the output is 'same', i.e. the
* output is the same size as the input.
*
*/

#include <iostream>
#include <cooperative_groups.h>
#include "masked_conv.hpp"

namespace cg = cooperative_groups;

__global__ void convolution_1d_kernel(
    float* out,
    float* arr,
    int n_arr,
    float* kernel,
    int n_kernel,
    bool* mask,
    float pad_val) {

    int t_idx{ threadIdx.x }; // thread index
    int b_dim{ blockDim.x }; // number of threads per block
    int b_idx{ blockIdx.x }; // block index

    int idx{ b_idx * b_dim + t_idx };
    int stride{ b_dim * gridDim.x }; // total number of threads

    for (int i = idx; i < n_arr; i += stride) {
        if (!mask[i]) continue; // skip if mask is false

        // loop over kernel
        for (int j = 0; j < n_kernel; ++j) {

            // index the array with implicit reversed kernel
            int input_index = i - j + n_kernel / 2;
            if (!mask[input_index]) continue; // skip if mask is false

            if (input_index >= 0 && input_index < n_arr) {
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
    convolution_1d_kernel << <grid_size, block_size >> > (
        out, arr, n_arr, kernel, n_kernel, mask, pad_val
        );
}