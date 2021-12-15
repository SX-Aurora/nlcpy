import cupy as cp
import time


############
#   2D     #
############

@cp.fuse
def cupy_2d_fusion_kernel_1(coef, x0, x1, x2, x3, x4, dout):
    dout[...] = (x0 + x1 + x2 + x3 + x4) * coef

@cp.fuse
def cupy_2d_fusion_kernel_2(coef, x0, x1, x2, x3, x4,
                            x5, x6, x7, x8,
                            dout):
    dout[...] = (x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8) * coef

@cp.fuse
def cupy_2d_fusion_kernel_4(coef, x0, x1, x2, x3, x4,
                            x5, x6, x7, x8, x9,
                            x10, x11, x12, x13, x14,
                            x15, x16,
                            dout):
    dout[...] = (x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 +
                 x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16) * coef

@cp.fuse
def cupy_2d_fusion_kernel_6(coef, x0, x1, x2, x3, x4,
                            x5, x6, x7, x8, x9,
                            x10, x11, x12, x13,
                            x14, x15, x16, x17,
                            x18, x19, x20, x21,
                            x22, x23, x24,
                            dout):
    dout[...] = (x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 +
                x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 +
                x17 + x18 + x19 + x20 + x21 + x22 + x23 + x24) * coef

def cupy_2d_fusion_impl(x, y, coef, N, I=1):

    cp.cuda.Device().synchronize()
    s = time.time()
    for i in range(I):
        if i % 2 == 0:
            din = x
            dout = y
        else:
            din = y
            dout = x
        if N == 1:
            cupy_2d_fusion_kernel_1(
                coef,
                din[..., 1:-1, 1:-1],
                din[..., 1:-1, 0:-2],
                din[..., 1:-1, 2:],
                din[..., 0:-2, 1:-1],
                din[..., 2:, 1:-1],
                dout[..., 1:-1, 1:-1])
        elif N == 2:
            cupy_2d_fusion_kernel_2(
                coef,
                din[..., 2:-2, 2:-2],
                din[..., 2:-2, 0:-4],
                din[..., 2:-2, 1:-3],
                din[..., 2:-2, 3:-1],
                din[..., 2:-2, 4:],
                din[..., 0:-4, 2:-2],
                din[..., 1:-3, 2:-2],
                din[..., 3:-1, 2:-2],
                din[..., 4:,   2:-2],
                dout[..., 2:-2, 2:-2])
        elif N == 4:
            cupy_2d_fusion_kernel_4(
                coef,
                din[..., 4:-4, 4:-4],
                din[..., 4:-4, 0:-8],
                din[..., 4:-4, 1:-7],
                din[..., 4:-4, 2:-6],
                din[..., 4:-4, 3:-5],
                din[..., 4:-4, 5:-3],
                din[..., 4:-4, 6:-2],
                din[..., 4:-4, 7:-1],
                din[..., 4:-4, 8:],
                din[..., 0:-8, 4:-4],
                din[..., 1:-7, 4:-4],
                din[..., 2:-6, 4:-4],
                din[..., 3:-5, 4:-4],
                din[..., 5:-3, 4:-4],
                din[..., 6:-2, 4:-4],
                din[..., 7:-1, 4:-4],
                din[..., 8:,   4:-4],
                dout[..., 4:-4, 4:-4])
        elif N == 6:
            cupy_2d_fusion_kernel_6(
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
    cp.cuda.Device().synchronize()
    e = time.time()
    return e - s, dout.get()


############
#   3D     #
############

@cp.fuse
def cupy_3d_fusion_kernel_1(coef, x0, x1, x2, x3, x4,
                            x5, x6, dout):
    dout[...] = (x0 + x1 + x2 + x3 + x4 + x5 + x6) * coef

@cp.fuse
def cupy_3d_fusion_kernel_2(coef, x0, x1, x2, x3, x4,
                            x5, x6, x7, x8, x9,
                            x10, x11, x12,
                            dout):
    dout[...] = (x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 +
                 x9 + x10 + x11 + x12) * coef

@cp.fuse
def cupy_3d_fusion_kernel_6(coef, x0, x1, x2, x3, x4,
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

def cupy_3d_fusion_impl(x, y, coef, N, I=1):

    cp.cuda.Device().synchronize()
    s = time.time()
    for i in range(I):
        if i % 2 == 0:
            din = x
            dout = y
        else:
            din = y
            dout = x
        if N == 1:
            cupy_3d_fusion_kernel_1(
                coef,
                din[..., 1:-1, 1:-1, 1:-1],
                din[..., 1:-1, 1:-1, 0:-2],
                din[..., 1:-1, 1:-1, 2:],
                din[..., 1:-1, 0:-2, 1:-1],
                din[..., 1:-1, 2:,   1:-1],
                din[..., 0:-2, 1:-1, 1:-1],
                din[..., 2:,   1:-1, 1:-1],
                dout[..., 1:-1, 1:-1, 1:-1])
        elif N == 2:
            cupy_3d_fusion_kernel_2(
                coef,
                din[..., 2:-2, 2:-2, 2:-2],
                din[..., 2:-2, 2:-2, 0:-4],
                din[..., 2:-2, 2:-2, 1:-3],
                din[..., 2:-2, 2:-2, 3:-1],
                din[..., 2:-2, 2:-2, 4:],
                din[..., 2:-2, 0:-4, 2:-2],
                din[..., 2:-2, 1:-3, 2:-2],
                din[..., 2:-2, 3:-1, 2:-2],
                din[..., 2:-2, 4:,   2:-2],
                din[..., 0:-4, 2:-2, 2:-2],
                din[..., 1:-3, 2:-2, 2:-2],
                din[..., 3:-1, 2:-2, 2:-2],
                din[..., 4:,   2:-2, 2:-2],
                dout[..., 2:-2, 2:-2, 2:-2])

        elif N == 6:
            cupy_3d_fusion_kernel_6(
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
    cp.cuda.Device().synchronize()
    e = time.time()
    return e - s, dout.get()
