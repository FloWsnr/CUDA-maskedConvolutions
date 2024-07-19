#pragma once

void convolution_1d(
    int grid_size,
    int block_size,
    float* out,
    float* arr,
    int n_arr,
    float* kernel,
    int n_kernel,
    bool* mask,
    float pad_val);