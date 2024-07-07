/*******************************************************************************
* Kernel to compute the masked 3D convolution of a tensor and a kernel.
* The convolution is done with padding and the output is 'same', i.e. the
* output is the same size as the input.
*
*/

#include <iostream>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void convolution_1d(
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