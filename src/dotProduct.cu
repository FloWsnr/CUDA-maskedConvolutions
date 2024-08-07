/*********************************************************************
 * Simple kernel to compute the dot product (skalar product) of two vectors
 ********************************************************************/
#include <iostream>
#include <chrono>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

void dot_product_cpu(float* out, float* v1, float* v2, int n) {
    for (int i = 0; i < n; i++) {
        *out += v1[i] * v2[i];
    }

}

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
    int vector_size = 10000000;
    /*********************************************************************
    ******************** CPU version
    ********************************************************************/
    // Declare variables
    float* v1{ new float[vector_size] };
    float* v2{ new float[vector_size] };
    float* v_out{ new float(0) };

    for (int i = 0; i < vector_size; i++) {
        v1[i] = 1.0f;
        v2[i] = 2.0f;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    dot_product_cpu(v_out, v1, v2, vector_size);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    std::cout << "CPU: Time measured: " << duration * 1e-6 << " seconds." << std::endl;
    std::cout << "Expected result:" << 2.0 * vector_size << std::endl;
    std::cout << "Result:" << *v_out << std::endl;

    // Free memory
    delete[] v1;
    delete[] v2;
    delete v_out;

    /*********************************************************************
     * ****************** GPU version
     * *******************************************************************/


     // Declare variables
    float* v3;
    float* v4;
    float* v_out_2;
    // TODO: use deviceQuery to get best block size
    int block_size = 256;
    // rounded up to nearest multiple of block size
    int num_blocks = (vector_size + block_size - 1) / block_size;

    // Allocate memory
    cudaMallocManaged(&v3, vector_size * sizeof(float));
    cudaMallocManaged(&v4, vector_size * sizeof(float));
    cudaMallocManaged(&v_out_2, sizeof(float));

    for (int i = 0; i < vector_size; i++) {
        v3[i] = 1.0f;
        v4[i] = 2.0f;
    }
    *v_out_2 = 0.0f;

    auto t3 = std::chrono::high_resolution_clock::now();
    // Launch kernel
    dot_product << < num_blocks, block_size >> > (v_out_2, v3, v4, vector_size);
    // Wait for kernel to finish
    cudaDeviceSynchronize();
    auto t4 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();
    std::cout << "GPU: Time measured: " << duration2 * 1e-6 << " seconds." << std::endl;
    // Print result
    std::cout << "Expected result:" << 2.0 * vector_size << std::endl;
    std::cout << "Result:" << *v_out_2 << std::endl;

    // Free memory
    cudaFree(v3);
    cudaFree(v4);
    cudaFree(v_out_2);

    return 0;
}