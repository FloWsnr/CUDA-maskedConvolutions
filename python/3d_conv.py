import scipy.ndimage as ndimage
import numpy as np

if __name__ == "__main__":
    arr = np.arange(1,28).reshape(3, 3, 3)
    kernel = np.ones((3, 3, 3))

    out = ndimage.convolve(arr, kernel, mode="constant", cval=0.0)
    out = out.flatten()

    print(out)