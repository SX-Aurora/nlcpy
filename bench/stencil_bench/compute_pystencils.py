import pystencils as ps
import numpy as np
try:
    import cupy as cp
    import pycuda
except ImportError:
    cp = None
    pycuda = None
import time


############
#   2D     #
############

def make_2d_update_rule_1(src, dst, coef):
    update_rule = [
        ps.Assignment(
            lhs=dst[0, 0],
            rhs=(
                src[ 0,  0] +
                src[ 0, -1] +
                src[ 0,  1] +
                src[-1,  0] +
                src[ 1,  0]
            ) * coef,
        )
    ]
    return update_rule

def make_2d_update_rule_6(src, dst, coef):
    update_rule = [
        ps.Assignment(
            lhs=dst[0, 0],
            rhs=(
                src[ 0,  0] +
                src[ 0, -6] +
                src[ 0, -5] +
                src[ 0, -4] +
                src[ 0, -3] +
                src[ 0, -2] +
                src[ 0, -1] +
                src[ 0,  6] +
                src[ 0,  5] +
                src[ 0,  4] +
                src[ 0,  3] +
                src[ 0,  2] +
                src[ 0,  1] +
                src[-6,  0] +
                src[-5,  0] +
                src[-4,  0] +
                src[-3,  0] +
                src[-2,  0] +
                src[-1,  0] +
                src[ 6,  0] +
                src[ 5,  0] +
                src[ 4,  0] +
                src[ 3,  0] +
                src[ 2,  0] +
                src[ 1,  0]
            ) * coef,
        )
    ]
    return update_rule

def pystencils_2d_cpu_impl(x, y, coef, N, I=1):
    if x.dtype == np.dtype('f4'):
        src, dst = ps.fields(
            'src, dst: float32[2D]',
            src=x, dst=y
        )
    elif x.dtype == np.dtype('f8'):
        src, dst = ps.fields(
            'src, dst: double[2D]',
            src=x, dst=y
        )
    else:
        raise TypeError

    if N == 1:
        update_rule = make_2d_update_rule_1(src, dst, coef)
    elif N == 6:
        update_rule = make_2d_update_rule_6(src, dst, coef)
    else:
        raise ValueError
    kernel = ps.create_kernel(update_rule, cpu_openmp=True).compile()

    s = time.time()
    for i in range(I):
        if i % 2 == 0:
            kernel(src=x, dst=y)
        else:
            kernel(src=y, dst=x)
    e = time.time()

    if (I - 1) % 2 == 0:
        res = y
    else:
        res = x
    return e - s, res

def pystencils_2d_gpu_impl(x, y, coef, N, I=1):
    if x.dtype == np.dtype('f4'):
        src, dst = ps.fields(
            'src, dst: float32[2D]',
            src=x, dst=y
        )
    elif x.dtype == np.dtype('f8'):
        src, dst = ps.fields(
            'src, dst: double[2D]',
            src=x, dst=y
        )
    else:
        raise TypeError

    if N == 1:
        update_rule = make_2d_update_rule_1(src, dst, coef)
    elif N == 6:
        update_rule = make_2d_update_rule_6(src, dst, coef)
    else:
        raise ValueError
    kernel = ps.create_kernel(update_rule, target='gpu').compile()

    x_gpu = pycuda.gpuarray.to_gpu(x)
    y_gpu = pycuda.gpuarray.to_gpu(y)

    pycuda.driver.Context.synchronize()
    s = time.time()
    for i in range(I):
        if i % 2 == 0:
            kernel(src=x_gpu, dst=y_gpu)
        else:
            kernel(src=y_gpu, dst=x_gpu)
    pycuda.driver.Context.synchronize()
    e = time.time()

    if (I - 1) % 2 == 0:
        res = y_gpu.get()
    else:
        res = x_gpu.get()
    return e - s, res


############
#   3D     #
############

def make_3d_update_rule_1(src, dst, coef):
    update_rule = [
        ps.Assignment(
            lhs=dst[0, 0, 0],
            rhs=(
                src[ 0,  0,  0] +
                src[ 0,  0, -1] +
                src[ 0,  0,  1] +
                src[ 0, -1,  0] +
                src[ 0,  1,  0] +
                src[-1,  0,  0] +
                src[ 1,  0,  0]
            ) * coef,
        )
    ]
    return update_rule

def make_3d_update_rule_6(src, dst, coef):
    update_rule = [
        ps.Assignment(
            lhs=dst[0, 0, 0],
            rhs=(
                src[ 0,  0,  0] +
                src[ 0,  0, -6] +
                src[ 0,  0, -5] +
                src[ 0,  0, -4] +
                src[ 0,  0, -3] +
                src[ 0,  0, -2] +
                src[ 0,  0, -1] +
                src[ 0,  0,  6] +
                src[ 0,  0,  5] +
                src[ 0,  0,  4] +
                src[ 0,  0,  3] +
                src[ 0,  0,  2] +
                src[ 0,  0,  1] +
                src[ 0, -6,  0] +
                src[ 0, -5,  0] +
                src[ 0, -4,  0] +
                src[ 0, -3,  0] +
                src[ 0, -2,  0] +
                src[ 0, -1,  0] +
                src[ 0,  6,  0] +
                src[ 0,  5,  0] +
                src[ 0,  4,  0] +
                src[ 0,  3,  0] +
                src[ 0,  2,  0] +
                src[ 0,  1,  0] +
                src[-6,  0,  0] +
                src[-5,  0,  0] +
                src[-4,  0,  0] +
                src[-3,  0,  0] +
                src[-2,  0,  0] +
                src[-1,  0,  0] +
                src[ 6,  0,  0] +
                src[ 5,  0,  0] +
                src[ 4,  0,  0] +
                src[ 3,  0,  0] +
                src[ 2,  0,  0] +
                src[ 1,  0,  0]
            ) * coef,
        )
    ]
    return update_rule


def pystencils_3d_cpu_impl(x, y, coef, N, I=1):
    if x.dtype == np.dtype('f4'):
        src, dst = ps.fields(
            'src, dst: float32[3D]',
            src=x, dst=y
        )
    elif x.dtype == np.dtype('f8'):
        src, dst = ps.fields(
            'src, dst: double[3D]',
            src=x, dst=y
        )
    else:
        raise TypeError

    if N == 1:
        update_rule = make_3d_update_rule_1(src, dst, coef)
    elif N == 6:
        update_rule = make_3d_update_rule_6(src, dst, coef)
    else:
        raise ValueError
    kernel = ps.create_kernel(update_rule, cpu_openmp=True).compile()

    s = time.time()
    for i in range(I):
        if i % 2 == 0:
            kernel(src=x, dst=y)
        else:
            kernel(src=y, dst=x)
    e = time.time()

    if (I - 1) % 2 == 0:
        res = y
    else:
        res = x
    return e - s, res


def pystencils_3d_gpu_impl(x, y, coef, N, I=1):
    if x.dtype == np.dtype('f4'):
        src, dst = ps.fields(
            'src, dst: float32[3D]',
            src=x, dst=y
        )
    elif x.dtype == np.dtype('f8'):
        src, dst = ps.fields(
            'src, dst: double[3D]',
            src=x, dst=y
        )
    else:
        raise TypeError

    if N == 1:
        update_rule = make_3d_update_rule_1(src, dst, coef)
    elif N == 6:
        update_rule = make_3d_update_rule_6(src, dst, coef)
    else:
        raise ValueError
    kernel = ps.create_kernel(update_rule, target='gpu').compile()

    x_gpu = pycuda.gpuarray.to_gpu(x)
    y_gpu = pycuda.gpuarray.to_gpu(y)

    pycuda.driver.Context.synchronize()
    s = time.time()
    for i in range(I):
        if i % 2 == 0:
            kernel(src=x_gpu, dst=y_gpu)
        else:
            kernel(src=y_gpu, dst=x_gpu)
    pycuda.driver.Context.synchronize()
    e = time.time()

    if (I - 1) % 2 == 0:
        res = y_gpu.get()
    else:
        res = x_gpu.get()
    return e - s, res
