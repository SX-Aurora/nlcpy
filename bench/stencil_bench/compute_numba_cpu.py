import numba
import numpy as np
import time


############
#   2D     #
############

@numba.jit(parallel=True)
def numba_2d_cpu_kernel_1(coef, x0, x1, x2, x3, x4, dout):
    dout[...] = (x0 + x1 + x2 + x3 + x4) * coef

@numba.jit(parallel=True)
def numba_2d_cpu_kernel_6(coef, x0, x1, x2, x3, x4,
                         x5, x6, x7, x8, x9,
                         x10, x11, x12, x13,
                         x14, x15, x16, x17,
                         x18, x19, x20, x21,
                         x22, x23, x24,
                         dout):
    dout[...] = (x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 +
                x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 +
                x17 + x18 + x19 + x20 + x21 + x22 + x23 + x24) * coef


@numba.njit
def numba_2d_launcher(x, y, coef, N, I=1):
    for i in range(I):
        if i % 2 == 0:
            din = x
            dout = y
        else:
            din = y
            dout = x
        if N == 1:
            numba_2d_cpu_kernel_1(
                coef,
                din[..., 1:-1, 1:-1],
                din[..., 1:-1, 0:-2],
                din[..., 1:-1, 2:],
                din[..., 0:-2, 1:-1],
                din[..., 2:, 1:-1],
                dout[..., 1:-1, 1:-1])
        elif N == 6:
            numba_2d_cpu_kernel_6(
                coef,
                din[..., 6:-6, 6:-6],
                din[..., 6:-6, 0:-12],
                din[..., 6:-6, 1:-11],
                din[..., 6:-6, 2:-10],
                din[..., 6:-6, 3:-9],
                din[..., 6:-6, 4:-8],
                din[..., 6:-6, 5:-7],
                din[..., 6:-6, 7:-5],
                din[..., 6:-6, 8:-4],
                din[..., 6:-6, 9:-3],
                din[..., 6:-6, 10:-2],
                din[..., 6:-6, 11:-1],
                din[..., 6:-6, 12:],
                din[..., 0:-12, 6:-6],
                din[..., 1:-11, 6:-6],
                din[..., 2:-10, 6:-6],
                din[..., 3:-9, 6:-6],
                din[..., 4:-8, 6:-6],
                din[..., 5:-7, 6:-6],
                din[..., 7:-5, 6:-6],
                din[..., 8:-4, 6:-6],
                din[..., 9:-3, 6:-6],
                din[..., 10:-2, 6:-6],
                din[..., 11:-1, 6:-6],
                din[..., 12:, 6:-6],
                dout[..., 6:-6, 6:-6])
        else:
            raise ValueError

    return dout


def numba_2d_impl(x, y, coef, N, I=1):
    coef = np.array(coef, dtype=x.dtype)
    # warmup
    _ = numba_2d_launcher(x, y, coef, N, I=1)

    s = time.time()
    res = numba_2d_launcher(x, y, coef, N, I=I)
    e = time.time()
    return e - s, res


############
#   3D     #
############

@numba.jit(parallel=True)
def numba_3d_cpu_kernel_1(coef, x0, x1, x2, x3, x4,
                         x5, x6, dout):
    dout[...] = (x0 + x1 + x2 + x3 + x4 + x5 + x6) * coef


@numba.jit(parallel=True)
def numba_3d_cpu_kernel_6(coef, x0, x1, x2, x3, x4,
                       x5, x6, x7, x8, x9,
                       x10, x11, x12, x13,
                       x14, x15, x16, x17,
                       x18, x19, x20, x21,
                       x22, x23, x24, x25,
                       x26, x27, x28, x29,
                       x30, x31, x32, x33,
                       x34, x35, x36,
                       dout):
    dout[...] = (x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 +
                 x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 +
                 x17 + x18 + x19 + x20 + x21 + x22 + x23 + x24 +
                 x25 + x26 + x27 + x28 + x29 + x30 + x31 + x32 +
                 x33 + x34 + x35 + x36) * coef


@numba.njit
def numba_3d_launcher(x, y, coef, N, I=1):
    for i in range(I):
        if i % 2 == 0:
            din = x
            dout = y
        else:
            din = y
            dout = x
        if N == 1:
            numba_3d_cpu_kernel_1(
                coef,
                din[..., 1:-1, 1:-1, 1:-1],
                din[..., 1:-1, 1:-1, 0:-2],
                din[..., 1:-1, 1:-1, 2:],
                din[..., 1:-1, 0:-2, 1:-1],
                din[..., 1:-1, 2:,   1:-1],
                din[..., 0:-2, 1:-1, 1:-1],
                din[..., 2:,   1:-1, 1:-1],
                dout[..., 1:-1, 1:-1, 1:-1])
        elif N == 6:
            numba_3d_cpu_kernel_6(
                coef,
                din[..., 6:-6,  6:-6,  6:-6],
                din[..., 6:-6,  6:-6,  0:-12],
                din[..., 6:-6,  6:-6,  1:-11],
                din[..., 6:-6,  6:-6,  2:-10],
                din[..., 6:-6,  6:-6,  3:-9],
                din[..., 6:-6,  6:-6,  4:-8],
                din[..., 6:-6,  6:-6,  5:-7],
                din[..., 6:-6,  6:-6,  7:-5],
                din[..., 6:-6,  6:-6,  8:-4],
                din[..., 6:-6,  6:-6,  9:-3],
                din[..., 6:-6,  6:-6,  10:-2],
                din[..., 6:-6,  6:-6,  11:-1],
                din[..., 6:-6,  6:-6,  12:],
                din[..., 6:-6,  0:-12, 6:-6],
                din[..., 6:-6,  1:-11, 6:-6],
                din[..., 6:-6,  2:-10, 6:-6],
                din[..., 6:-6,  3:-9,  6:-6],
                din[..., 6:-6,  4:-8,  6:-6],
                din[..., 6:-6,  5:-7,  6:-6],
                din[..., 6:-6,  7:-5,  6:-6],
                din[..., 6:-6,  8:-4,  6:-6],
                din[..., 6:-6,  9:-3,  6:-6],
                din[..., 6:-6,  10:-2, 6:-6],
                din[..., 6:-6,  11:-1, 6:-6],
                din[..., 6:-6,  12:, 6:-6],
                din[..., 0:-12, 6:-6, 6:-6],
                din[..., 1:-11, 6:-6, 6:-6],
                din[..., 2:-10, 6:-6, 6:-6],
                din[..., 3:-9,  6:-6, 6:-6],
                din[..., 4:-8,  6:-6, 6:-6],
                din[..., 5:-7,  6:-6, 6:-6],
                din[..., 7:-5,  6:-6, 6:-6],
                din[..., 8:-4,  6:-6, 6:-6],
                din[..., 9:-3,  6:-6, 6:-6],
                din[..., 10:-2, 6:-6, 6:-6],
                din[..., 11:-1, 6:-6, 6:-6],
                din[..., 12:,   6:-6, 6:-6],
                dout[..., 6:-6, 6:-6, 6:-6])
        else:
            raise ValueError

    return dout

def numba_3d_impl(x, y, coef, N, I=1):
    coef = np.array(coef, dtype=x.dtype)
    # warmup
    _ = numba_3d_launcher(x, y, coef, N, I=1)

    s = time.time()
    res = numba_3d_launcher(x, y, coef, N, I=I)
    e = time.time()
    return e - s, res
