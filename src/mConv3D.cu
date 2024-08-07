
/*****************************************************************
 * 3D convolution by using special strides in 1D memory
 *
 * The 3D grid layout of CUDA is NOT used, since cupy calling
 * this kernel is working with 1D data.
 *****************************************************************/
#include <iostream>
#include <cooperative_groups.h>
#include "mConv.hpp"

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
    const unsigned int t_idx{ threadIdx.x }; // thread index
    const unsigned int b_dim{ blockDim.x }; // number of threads per block
    const unsigned int b_idx{ blockIdx.x }; // block index

    const unsigned int idx{ b_idx * b_dim + t_idx };
    const unsigned int stride{ b_dim * gridDim.x }; // total number of threads

    const int n_arr = nx_arr * ny_arr * nz_arr;
    const int n_kernel = nx_kernel * ny_kernel * nz_kernel;


    int i, j, x, y, z;
    int xShift, yShift, zShift;
    int input_index, input_index_x, input_index_y, input_index_z;
    const int kernelCenterX = nx_kernel / 2;
    const int kernelCenterY = ny_kernel / 2;
    const int kernelCenterZ = nz_kernel / 2;

    for (i = idx; i < n_arr; i += stride) {
        if (!mask[i]) continue; // skip if mask is false

        // loop over kernel
        for (x = 0; x < nx_kernel; ++x) {
            for (y = 0; y < ny_kernel; ++y) {
                for (z = 0; z < nz_kernel; ++z) {

                    xShift = x - kernelCenterX;
                    yShift = y - kernelCenterY;
                    zShift = z - kernelCenterZ;

                    input_index_x = (x - xShift) / 2;
                    input_index_y = y - yShift / 2;
                    input_index_z = z - zShift / 2;

                    j = input_index_x * ny_kernel * nz_kernel + input_index_y * nz_kernel + input_index_z;
                    input_index = i - j + n_kernel / 2;
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