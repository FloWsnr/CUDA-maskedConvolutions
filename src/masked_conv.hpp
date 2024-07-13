#pragma once

void convolution_1d(float* out,
    float* arr,
    int n_arr,
    float* kernel,
    int n_kernel,
    bool* mask,
    float pad_val);