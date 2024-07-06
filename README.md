# CUDA-maskedConvolutions
Custom CUDA kernel to accelerate masked 3D convolutions used as raw kernel in cupy.
The source can then be imported into python.

Currently used to learn CUDA, hence the code is not production-ready and might not even be correct yet. Please use at your own risk.


Notes:
- Thrust lib is awesome to use std-like features (vector, unique_ptr etc)