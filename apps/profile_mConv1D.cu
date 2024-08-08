#include "mConv.hpp"

int main() {

    int vector_size = 10000;
    int kernel_size = 500;

    int block_size = 256;
    int num_blocks = 4;

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

        // Set every other element of mask to true
        if (i % 2 == 0) {
            mask[i] = true;
        }
        else {
            mask[i] = false;
        }
    }
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] = 4.0f + i;
    }

    // Launch kernel
    convolution_1d(num_blocks, block_size, v_out, v1, vector_size, kernel, kernel_size, mask, 0.0f);
    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Free memory
    cudaFree(v1);
    cudaFree(kernel);
    cudaFree(v_out);
    cudaFree(mask);
}
