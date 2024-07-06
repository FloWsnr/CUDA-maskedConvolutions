/*******************************************************************************
* Kernel to compute the masked 3D convolution of a tensor and a kernel.
* The convolution is done with padding and the output is 'same', i.e. the
* output is the same size as the input.
*
*/

#include <iostream>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;


__global__ void dot_product(float* out, float* v1, float* v2, int n) {

    cg::thread_block cta = cg::this_thread_block();

    int t_idx = threadIdx.x; // thread index
    int b_dim = blockDim.x; // number of threads per block
    int b_idx = blockIdx.x; // block index

    int idx = b_idx * b_dim + t_idx;
    int stride = b_dim * gridDim.x; // total number of threads

    // make shared memory the next power of 2 multiple of block size
    extern __shared__ float sdata[]; // shared memory for intermediate results


    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        // each thread computes one element (or multiple)
        // of the output and already computes the sum of these elements
        sum += v1[i] * v2[i];
    }
    sdata[t_idx] = sum; // partial sum is stored in shared memory
    cg::sync(cta); // wait for all threads to finish

    // reduction
    for (int stride = b_dim / 2; stride > 0; stride /= 2) {
        // the lower half of threads per block perform the reduction
        // afterwards, the stride is divided by two and repeat
        // until all values are summed in element 0 of the shared memory

        // Example:
        // block dim: 256
        // -> stride: 128
        // -> sdata[0] = sdata[0] + sdata[128]
        // -> sdata[1] = sdata[1] + sdata[129]

        // Next iteration:
        // -> stride: 64
        // -> sdata[0] = sdata[0] + sdata[64]
        // -> sdata[1] = sdata[1] + sdata[65]

        if (t_idx < stride) {
            sdata[t_idx] += sdata[t_idx + stride];
        }
        cg::sync(cta); // wait for all threads to finish
    }

    if (t_idx == 0) {
        atomicAdd(out, sdata[0]);
    }
}

__global__ void convolution_1d(
    float* out,
    float* arr,
    int n_arr,
    float* kernel,
    int n_kernel,
    bool* mask,
    float pad_val) {

    int t_idx = threadIdx.x; // thread index
    int b_dim = blockDim.x; // number of threads per block
    int b_idx = blockIdx.x; // block index

    int idx = b_idx * b_dim + t_idx;
    int stride = b_dim * gridDim.x; // total number of threads

    for (int i = idx; i < n_arr; i += stride) {
        out[i] = 0.0f;

        // loop over kernel
        for (int j = 0; j < n_kernel; ++j) {

            // index the array with implicitreversed kernel
            int input_index = i - j + n_kernel / 2;

            if (input_index >= 0 && input_index < n_arr) {
                out[i] += arr[input_index] * kernel[j];
            }
            else {
                out[i] += pad_val * kernel[j];
            }
        }
    }
}

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


    // Initialize variables
    // for (int i = 0; i < vector_size; i++) {
    //     v1[i] = 4.0f;
    //     if (i % 2 == 0) {
    //         mask[i] = true;
    //     }
    //     else {
    //         mask[i] = false;
    //     }
    // }

    // for (int i = 0; i < kernel_size; i++) {
    //     kernel[i] = 1.0f / kernel_size;
    // }

    // Launch kernel
    convolution_1d << < num_blocks, block_size >> > (v_out, v1, vector_size, kernel, kernel_size, mask, 0.0f);

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