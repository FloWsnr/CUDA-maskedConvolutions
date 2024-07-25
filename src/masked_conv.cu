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

    unsigned int t_idx{ threadIdx.x }; // thread index
    unsigned int b_dim{ blockDim.x }; // number of threads per block
    unsigned int b_idx{ blockIdx.x }; // block index

    unsigned int idx{ b_idx * b_dim + t_idx };
    unsigned int stride{ b_dim * gridDim.x }; // total number of threads

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

/*****************************************************************
 * 3D convolution by using special strides in 1D memory
 *
 * The 3D grid layout of CUDA is NOT used, since cupy calling
 * this kernel is working with 1D data.
 *****************************************************************/
__global__ void convolution_3d_kernel(
    float* out,
    const float* arr,
    const int nx_arr,
    const int ny_arr,
    const int nz_arr,
    const float* kernel,
    const int nx_kernel,
    const int ny_kernel,
    const int nz_kernel,
    const bool* mask,
    const float pad_val) {

    // TODO: use shared memory for the parameters

    int n_arr = nx_arr * ny_arr * nz_arr;
    int n_kernel = nx_kernel * ny_kernel * nz_kernel;

    const unsigned int t_idx{ threadIdx.x }; // thread index
    const unsigned int b_dim{ blockDim.x }; // number of threads per block
    const unsigned int b_idx{ blockIdx.x }; // block index

    const unsigned int idx{ b_idx * b_dim + t_idx };
    const unsigned int stride{ b_dim * gridDim.x }; // total number of threads

    for (int i = idx; i < n_arr; i += stride) {
        if (!mask[i]) continue; // skip if mask is false

        // loop over kernel
        for (int x = 0; x < nx_kernel; ++x) {
            for (int y = 0; y < ny_kernel; ++y) {
                for (int z = 0; z < nz_kernel; ++z) {
                    int j = x * ny_kernel * nz_kernel + y * nz_kernel + z;
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
    }
}

void convolution_3d(
    int grid_size,
    int block_size,
    float* out,
    float* arr,
    int nx_arr,
    int ny_arr,
    int nz_arr,
    float* kernel,
    int nx_kernel,
    int ny_kernel,
    int nz_kernel,
    bool* mask,
    float pad_val) {
    convolution_3d_kernel << <grid_size, block_size >> > (
        out, arr, nx_arr, ny_arr, nz_arr, kernel, nx_kernel, ny_kernel, nz_kernel, mask, pad_val
        );
}