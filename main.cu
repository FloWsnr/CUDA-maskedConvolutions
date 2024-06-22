#include <iostream>

__global__ void add_one(int& x) {
    x += 1;
}

__global__ void vector_product(float* out, float* v1, float* v2, int n) {

    int t_idx = threadIdx.x; // thread index
    int b_dim = blockDim.x; // number of threads per block
    int b_idx = blockIdx.x; // block index

    int idx = b_idx * b_dim + t_idx;
    int stride = b_dim * gridDim.x; // total number of threads

    for (int i = idx; i < n; i += stride) {
        out[i] = v1[i] * v2[i];
    }
}

int main() {
    int N = 1000000;
    int block_size = 256;
    // rounded up to nearest multiple of block size
    int num_blocks = (N + block_size - 1) / block_size;

    // Declare variables
    float* v1;
    float* v2;
    float* out;

    // Allocate memory
    cudaMallocManaged(&v1, N * sizeof(float));
    cudaMallocManaged(&v2, N * sizeof(float));
    cudaMallocManaged(&out, N * sizeof(float));

    // Initialize variables
    for (int i = 0; i < N; i++) {
        v1[i] = i;
        v2[i] = i;
    }

    // Launch kernel
    vector_product << < num_blocks, block_size >> > (out, v1, v2, N);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Print result
    std::cout << out[0] << std::endl;


    // Free memory
    cudaFree(v1);
    cudaFree(v2);
    cudaFree(out);

    return 0;
}