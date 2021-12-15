import numba
from numba import cuda
import time
import math


############
#   2D     #
############

@cuda.jit
def numba_2d_cuda_kernel_1(x, coef, out):
    j, i = cuda.grid(2)
    ny, nx = x.shape
    if 1 <= i < ny - 1 and 1 <= j < nx - 1:
        out[i, j] = (x[i    , j    ] +
                     x[i    , j - 1] +
                     x[i    , j + 1] +
                     x[i - 1, j    ] +
                     x[i + 1, j    ]) * coef

@cuda.jit
def numba_2d_cuda_kernel_6(x, coef, out):
    j, i = cuda.grid(2)
    ny, nx = x.shape
    if 6 <= i < ny - 6 and 6 <= j < nx - 6:
        out[i, j] = (x[i    , j    ] +
                     x[i    , j - 6] +
                     x[i    , j - 5] +
                     x[i    , j - 4] +
                     x[i    , j - 3] +
                     x[i    , j - 2] +
                     x[i    , j - 1] +
                     x[i    , j + 1] +
                     x[i    , j + 2] +
                     x[i    , j + 3] +
                     x[i    , j + 4] +
                     x[i    , j + 5] +
                     x[i    , j + 6] +
                     x[i - 6, j    ] +
                     x[i - 5, j    ] +
                     x[i - 4, j    ] +
                     x[i - 3, j    ] +
                     x[i - 2, j    ] +
                     x[i - 1, j    ] +
                     x[i + 1, j    ] +
                     x[i + 2, j    ] +
                     x[i + 3, j    ] +
                     x[i + 4, j    ] +
                     x[i + 5, j    ] +
                     x[i + 6, j    ]) * coef


def numba_2d_cuda_impl(x, y, coef, N, I=1):

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(y.shape[-1] / threadsperblock[-1])
    blockspergrid_y = math.ceil(y.shape[-2] / threadsperblock[-2])
    blockspergrid = (
        blockspergrid_x,
        blockspergrid_y,
    )

    numba.cuda.synchronize()
    s = time.time()
    for i in range(I):
        if N == 1:
            if i % 2 == 0:
                numba_2d_cuda_kernel_1[blockspergrid, threadsperblock](x, coef, y)
            else:
                numba_2d_cuda_kernel_1[blockspergrid, threadsperblock](y, coef, x)
        elif N == 6:
            if i % 2 == 0:
                numba_2d_cuda_kernel_6[blockspergrid, threadsperblock](x, coef, y)
            else:
                numba_2d_cuda_kernel_6[blockspergrid, threadsperblock](y, coef, x)
        else:
            raise ValueError
    numba.cuda.synchronize()
    e = time.time()
    if (I - 1) % 2 == 0:
        res = y
    else:
        res = x
    return e - s, res.get()


############
#   3D     #
############

@cuda.jit
def numba_3d_cuda_kernel_1(x, coef, out):
    k, j, i = cuda.grid(3)
    nz, ny, nx = x.shape
    if 1 <= i < nz - 1 and 1 <= j < ny - 1 and 1 <= k < nx - 1:
        out[i, j, k] = (x[i    , j    , k    ] +
                        x[i - 1, j    , k    ] +
                        x[i + 1, j    , k    ] +
                        x[i    , j - 1, k    ] +
                        x[i    , j + 1, k    ] +
                        x[i    , j    , k - 1] +
                        x[i    , j    , k + 1]
                    ) * coef

@cuda.jit
def numba_3d_cuda_kernel_2(x, coef, out):
    k, j, i = cuda.grid(3)
    nz, ny, nx = x.shape
    if 2 <= i < nz - 2 and 2 <= j < ny - 2 and 2 <= k < nx - 2:
        out[i, j, k] = (x[i    , j    , k    ] +
                        x[i - 2, j    , k    ] +
                        x[i - 1, j    , k    ] +
                        x[i + 2, j    , k    ] +
                        x[i + 1, j    , k    ] +
                        x[i    , j - 2, k    ] +
                        x[i    , j - 1, k    ] +
                        x[i    , j + 2, k    ] +
                        x[i    , j + 1, k    ] +
                        x[i    , j    , k - 2] +
                        x[i    , j    , k - 1] +
                        x[i    , j    , k + 2] +
                        x[i    , j    , k + 1]
                    ) * coef

@cuda.jit
def numba_3d_cuda_kernel_6(x, coef, out):
    k, j, i = cuda.grid(3)
    nz, ny, nx = x.shape
    if 6 <= i < nz - 6 and 6 <= j < ny - 6 and 6 <= k < nx - 6:
        out[i, j, k] = (x[i    , j    , k    ] +
                        x[i - 6, j    , k    ] +
                        x[i - 5, j    , k    ] +
                        x[i - 4, j    , k    ] +
                        x[i - 3, j    , k    ] +
                        x[i - 2, j    , k    ] +
                        x[i - 1, j    , k    ] +
                        x[i + 6, j    , k    ] +
                        x[i + 5, j    , k    ] +
                        x[i + 4, j    , k    ] +
                        x[i + 3, j    , k    ] +
                        x[i + 2, j    , k    ] +
                        x[i + 1, j    , k    ] +
                        x[i    , j - 6, k    ] +
                        x[i    , j - 5, k    ] +
                        x[i    , j - 4, k    ] +
                        x[i    , j - 3, k    ] +
                        x[i    , j - 2, k    ] +
                        x[i    , j - 1, k    ] +
                        x[i    , j + 6, k    ] +
                        x[i    , j + 5, k    ] +
                        x[i    , j + 4, k    ] +
                        x[i    , j + 3, k    ] +
                        x[i    , j + 2, k    ] +
                        x[i    , j + 1, k    ] +
                        x[i    , j    , k - 6] +
                        x[i    , j    , k - 5] +
                        x[i    , j    , k - 4] +
                        x[i    , j    , k - 3] +
                        x[i    , j    , k - 2] +
                        x[i    , j    , k - 1] +
                        x[i    , j    , k + 6] +
                        x[i    , j    , k + 5] +
                        x[i    , j    , k + 4] +
                        x[i    , j    , k + 3] +
                        x[i    , j    , k + 2] +
                        x[i    , j    , k + 1]
                    ) * coef


def numba_3d_cuda_impl(x, y, coef, N, I=1):

    threadsperblock = (8, 8, 8)
    blockspergrid_x = math.ceil(y.shape[-1] / threadsperblock[-1])
    blockspergrid_y = math.ceil(y.shape[-2] / threadsperblock[-2])
    blockspergrid_z = math.ceil(y.shape[-3] / threadsperblock[-3])
    blockspergrid = (
        blockspergrid_x,
        blockspergrid_y,
        blockspergrid_z,
    )

    numba.cuda.synchronize()
    s = time.time()
    for i in range(I):
        if N == 1:
            if i % 2 == 0:
                numba_3d_cuda_kernel_1[blockspergrid, threadsperblock](x, coef, y)
            else:
                numba_3d_cuda_kernel_1[blockspergrid, threadsperblock](y, coef, x)
        elif N == 2:
            if i % 2 == 0:
                numba_3d_cuda_kernel_2[blockspergrid, threadsperblock](x, coef, y)
            else:
                numba_3d_cuda_kernel_2[blockspergrid, threadsperblock](y, coef, x)
        elif N == 6:
            if i % 2 == 0:
                numba_3d_cuda_kernel_6[blockspergrid, threadsperblock](x, coef, y)
            else:
                numba_3d_cuda_kernel_6[blockspergrid, threadsperblock](y, coef, x)
        else:
            raise ValueError
    numba.cuda.synchronize()
    e = time.time()
    if (I - 1) % 2 == 0:
        res = y
    else:
        res = x
    return e - s, res.get()
