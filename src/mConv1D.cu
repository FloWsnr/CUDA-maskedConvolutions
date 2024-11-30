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

__global__ void convolution_1d_kernel(
    float* out,
    const float* __restrict__ arr,
    const int n_arr,
    const float* __restrict__ kernel,
    const int n_kernel,
    const bool* __restrict__ mask,
    const float pad_val) {

    const unsigned int t_idx{ threadIdx.x }; // thread index
    const unsigned int b_dim{ blockDim.x }; // number of threads per block
    const unsigned int b_idx{ blockIdx.x }; // block index

    const unsigned int idx{ b_idx * b_dim + t_idx };
    const unsigned int stride{ b_dim * gridDim.x }; // total number of threads

    const unsigned int k_radius = n_kernel / 2;
    // Shared memory
    /*******************************************************
    First partion for array data is as large as
    the number of threads (blockDim.x) so every thread has one
    element to work with. The rest is for the kernel.
    *******************************************************/
    extern __shared__ float shared_mem[];
    float* s_arr = shared_mem;
    float* s_kernel = &shared_mem[blockDim.x];


    // Global index for this block to start from
    // 2 * radius are the halo regions which are loaded but not computed by this block
    const int block_start = b_idx * (b_dim - 2 * k_radius);
    const int halo_start = block_start - k_radius;

    // Load input array into shared memory
    if (t_idx < b_dim) {

        /*
        each thread gets an index of the global array:
        The index is the block start index for this block plus
        the thread id in this block
        */
        int load_idx = halo_start + t_idx;
        if (load_idx >= 0 && load_idx < n_arr) {

            // s_arr is new for each block, thus we index it only by the
            // block thread_id. The array must be indexed by the global index
            s_arr[t_idx] = arr[load_idx];
        }
        else { s_arr[t_idx] = pad_val; }
    }

    // Load kernel into shared memory
    // strided loop to account for a kernel which is bigger than
    // the b_dim (number of threads)
    for (int i = t_idx; i < n_kernel; i += b_dim) {
        s_kernel[i] = kernel[i];
    }

    // Create a cooperative group for the block
    cg::sync(cg::this_thread_block());

    // Compute only for threads that are not in the halo regions
    if (t_idx >= k_radius && t_idx < b_dim - k_radius) {

        // since thread_id only starts at k_radius, we substract radius again
        // thus, t_idx = radius computes the first element (blockstart) and
        // t_idx + 1 computes the second ...
        int out_idx = block_start + t_idx - k_radius;
        if (out_idx < n_arr) {
            if (!mask[out_idx]) {
                out[out_idx] = pad_val;
            }
            else {
                float sum = 0.0f;

                for (int j = 0; j < n_kernel; ++j) {
                    // index the shared array data but in reverse order
                    int s_arr_idx = t_idx + k_radius - j;

                    // make sure we only index shared data in the first b_dim elements
                    if (s_arr_idx >= 0 && s_arr_idx < b_dim) {

                        // global index
                        int g_idx = out_idx + k_radius - j;
                        if (g_idx < 0 || g_idx >= n_arr || mask[g_idx]) {
                            sum += s_arr[s_arr_idx] * s_kernel[j];
                        }
                        else {
                            sum += pad_val * s_kernel[j];
                        }
                    }
                }

                out[out_idx] = sum;
            }
        }
    }
}

void convolution_1d(
    const int block_size,
    float* out,
    const float* arr,
    const int n_arr,
    const float* kernel,
    const int n_kernel,
    const bool* mask,
    const float pad_val) {

    const int shared_mem_size = (block_size + n_kernel) * sizeof(float);

    // only some threads actually compute an element, i.e. the center threads
    // the halo-threads, kernel radius (n_kernel /2) on both sides, do not.
    const int effective_block_size = block_size - 2 * (n_kernel / 2);

    // common cuda idiom to calculate the number of blocks
    const int grid_size = (n_arr + effective_block_size - 1) / effective_block_size;

    convolution_1d_kernel << <grid_size, block_size, shared_mem_size >> > (
        out, arr, n_arr, kernel, n_kernel, mask, pad_val
        );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch convolution_1d_kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void convolution_1d_cpu(
    float* out,
    const float* arr,
    const int n_arr,
    const float* kernel,
    const int n_kernel,
    const bool* mask,
    const float pad_val) {

    const int k_radius = n_kernel / 2;  // Precalculate kernel radius

    for (int i = 0; i < n_arr; i++) {
        if (!mask[i]) continue; // skip if mask is false

        // loop over kernel
        float sum = 0.0f;
        const int start_idx = i + k_radius;
        for (int j = 0; j < n_kernel; ++j) {

            // index the array with implicit reversed kernel
            int input_index = start_idx - j;
            if (input_index >= 0 && input_index < n_arr) {
                if (!mask[input_index]) continue; // skip if mask is false
                sum += arr[input_index] * kernel[j];
            }
            else {
                sum += pad_val * kernel[j];
            }
        }
        out[i] = sum;
    }
}