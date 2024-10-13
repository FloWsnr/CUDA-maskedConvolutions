#include <chrono>
#include <iostream>
#include <functional>
#include "mConv.hpp"

auto benchmark_cpu(int vector_size, int kernel_size, int num_trials) {

    // Declare variables
    float* v1{ new float[vector_size] };
    float* kernel{ new float[kernel_size] };
    float* v_out{ new float[vector_size] };
    bool* mask{ new bool[vector_size] };

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

    auto total_duration = std::chrono::nanoseconds::zero();
    // Launch
    for (int i = 0; i < num_trials; i++) {

        auto start = std::chrono::high_resolution_clock::now();
        convolution_1d_cpu(v_out, v1, vector_size, kernel, kernel_size, mask, 0.0f);
        // Record time
        auto end = std::chrono::high_resolution_clock::now();
        total_duration += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    }

    auto duration = total_duration.count() / num_trials;

    delete[] v1;
    delete[] kernel;
    delete[] v_out;
    delete[] mask;

    return duration;
}

auto benchmark_gpu(int vector_size, int kernel_size, int num_trials) {
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

    auto total_duration = std::chrono::nanoseconds::zero();
    // Launch
    for (int i = 0; i < num_trials; i++) {

        auto start = std::chrono::high_resolution_clock::now();
        // Launch kernel
        convolution_1d(v_out, v1, vector_size, kernel, kernel_size, mask, 0.0f);
        // Wait for kernel to finish
        cudaDeviceSynchronize();
        // Record time
        auto end = std::chrono::high_resolution_clock::now();
        total_duration += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    }
    auto duration = total_duration.count() / num_trials;

    cudaFree(v1);
    cudaFree(kernel);
    cudaFree(v_out);
    cudaFree(mask);

    return duration;
}


int main() {


    const int num_trials = 100;
    int vector_size = 100024;
    int kernel_size = 100;

    // Max concurrent threads = SM Count * threads per SM
    // = 68 * 1536 (for RTX 3080)

    auto cpu_duration = benchmark_cpu(vector_size, kernel_size, num_trials);
    std::cout << "CPU: Average time measured: " << cpu_duration << " nanoseconds." << std::endl;

    auto gpu_duration = benchmark_gpu(vector_size, kernel_size, num_trials);
    std::cout << "GPU: Average time measured: " << gpu_duration << " nanoseconds." << std::endl;
}