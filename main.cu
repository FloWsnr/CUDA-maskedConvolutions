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

int main() {
    int N = 10000000;
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
        v1[i] = 1.0f;
        v2[i] = 2.0f;
    }

    // Launch kernel
    dot_product << < num_blocks, block_size >> > (out, v1, v2, N);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Print result
    // should be 2 * N
    std::cout << *out << std::endl;


    // Free memory
    cudaFree(v1);
    cudaFree(v2);
    cudaFree(out);

    return 0;
}