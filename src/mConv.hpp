#pragma once

void convolution_1d(
    float* out,
    const float* arr,
    const int n_arr,
    const float* kernel,
    const int n_kernel,
    const bool* mask,
    const float pad_val);


void convolution_3d(
    int grid_size,
    int block_size,
    float* out,
    float* arr,
    int nx_arr,
    int ny_arr,
    int nz_arr,
    float* kernel,
    int nx_kernel,
    int ny_kernel,
    int nz_kernel,
    bool* mask,
    float pad_val);

void convolution_1d_cpu(
    float* out,
    const float* arr,
    const int n_arr,
    const float* kernel,
    const int n_kernel,
    const bool* mask,
    const float pad_val);
