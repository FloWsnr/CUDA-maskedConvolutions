# CUDA - masked convolutions
Custom CUDA kernel to accelerate masked 3D convolutions used as raw kernel in cupy.
Currently used to learn CUDA, hence the code is not production-ready and might not even be correct yet. Please use at your own risk.

## Motivation

Masked convolutions are convolutions where the input array contain "invalid" elements which should not participate in the end result. For example, consider an array with 3 elements and a normalized kernel with 3 elements. Lets assume that the central element in the input array is invalid, i.e. the mask is [0,1,1].
```
arrary = [1,2,1]
kernel = [0.25,0.5,0.25]
mask = [0,1,1]
```

### Normal masked convolution (zero padding):
```
array * kernel =
[
First element: invalid (since mask is false)
Second element: 0.25 * invalid + 0.5 * 2 + 0.25 * 1 = 1.25
Third element:  0.25 * 2 + 0.5 * 1 + 0 (zero padding) * 0.25  = 1.0
]
=> [invalid, 1.25, 1.0]
```

Additionally, in some usecases (or maybe just in mine), the convolution with a normalized kernel should always result in a normalized output, i.e. the kernel weight which participated in the computation of an element should sum up to 1, **even if some input elements are invalid**. In that case, the masked convolution gets more complex:

### Normalized masked convolution (zero padding):
```
First element: invalid (since mask is false)

Second element: 0.25 * invalid + 0.5 * 2 + 0.25 * 1
    Only two kernel elements are "active", i.e. placed on valid array elements.
    Thus, these kernel elements are scaled by their sum to again be 1. (0.5/0.75, 0.25/0.75)
    Thus the calculation becomes: 0.66 * 2 + 0.33 * 1 = 1.66

Third element:  0.25 * 2 + 0.5 * 1 + 0 (zero padding) * 0.25
    Similar calculations are performed for zero padding.
    0.33 * 2 + 0.66 * 1 = 1.32
}
=> [invalid, 1.66, 1.32]
```

You can read more about that usecase in the following [publication](https://pubs.acs.org/doi/abs/10.1021/acsami.4c04641)


## Usage
The source code can be used as a [raw kernel](https://docs.cupy.dev/en/stable/user_guide/kernel.html#raw-kernels) in cupy.
Alternatively, the kernels can be compiled and then used as [raw modules](https://docs.cupy.dev/en/stable/user_guide/kernel.html#raw-modules) in cupy.

## Notes:
- Thrust lib is awesome to use std-like features (vector, unique_ptr etc)