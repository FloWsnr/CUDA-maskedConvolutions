#include <iostream>
#include <gtest/gtest.h>
#include "mConv.hpp"

TEST(Conv1D_CPU, TestNormalConvWorks) {
    int vector_size = 3;
    int kernel_size = 3;

    // Declare variables
    float* v1 = new float[vector_size];
    float* kernel = new float[kernel_size];
    float* v_out = new float[vector_size];
    bool* mask = new bool[vector_size];

    for (int i = 0; i < vector_size; i++) {
        v1[i] = i + 1.0f;
        v_out[i] = 0.0f;
        mask[i] = true;
    }
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] = 4.0f + i;
    }

    // Launch kernel
    convolution_1d_cpu(v_out, v1, vector_size, kernel, kernel_size, mask, 0.0f);

    // Check results
    EXPECT_EQ(v_out[0], 13.0f);
    EXPECT_EQ(v_out[1], 28.0f);
    EXPECT_EQ(v_out[2], 27.0f);

    // Free memory
    delete[] v1;
    delete[] kernel;
    delete[] v_out;
    delete[] mask;
}