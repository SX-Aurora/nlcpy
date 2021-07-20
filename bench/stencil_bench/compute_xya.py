from __future__ import print_function
import numpy as np
import nlcpy as vp
import numba
from math import *
import time

# target libraries
nb = 'numba'
vp_naive = 'nlcpy_naive'
vp_sca = 'nlcpy_sca'


@numba.stencil
def numba_kernel_1(din):
    return (din[0, 0, -1] +
            din[0, 0, 0] +
            din[0, 0, 1] +
            din[0, -1, 0] +
            din[0, 1, 0]
            )

@numba.stencil
def numba_kernel_2(din):
    return (din[0, 0, -2] +
            din[0, 0, -1] +
            din[0, 0, 0] +
            din[0, 0, 1] +
            din[0, 0, 2] +
            din[0, -2, 0] +
            din[0, -1, 0] +
            din[0, 2, 0] +
            din[0, 1, 0]
            )

@numba.stencil
def numba_kernel_3(din):
    return (din[0, 0, -3] +
            din[0, 0, -2] +
            din[0, 0, -1] +
            din[0, 0, 0] +
            din[0, 0, 1] +
            din[0, 0, 2] +
            din[0, 0, 3] +
            din[0, -3, 0] +
            din[0, -2, 0] +
            din[0, -1, 0] +
            din[0, 3, 0] +
            din[0, 2, 0] +
            din[0, 1, 0]
            )

@numba.stencil
def numba_kernel_4(din):
    return (din[0, 0, -4] +
            din[0, 0, -3] +
            din[0, 0, -2] +
            din[0, 0, -1] +
            din[0, 0, 0] +
            din[0, 0, 1] +
            din[0, 0, 2] +
            din[0, 0, 3] +
            din[0, 0, 4] +
            din[0, -4, 0] +
            din[0, -3, 0] +
            din[0, -2, 0] +
            din[0, -1, 0] +
            din[0, 4, 0] +
            din[0, 3, 0] +
            din[0, 2, 0] +
            din[0, 1, 0]
            )

@numba.njit
def numba_launcher(din, dout, N, I=1):
    for _ in range(I):
        if N == 1:
            numba_kernel_1(din, out=dout)
        elif N == 2:
            numba_kernel_2(din, out=dout)
        elif N == 3:
            numba_kernel_3(din, out=dout)
        elif N == 4:
            numba_kernel_4(din, out=dout)

def numba_impl(din, dout, N, I=1):
    # warmup
    numba_launcher(din, dout, N, I=1)

    s = time.time()
    numba_launcher(din, dout, N, I=I)
    e = time.time()
    return e - s


def nlcpy_naive_impl(din, dout, N, I=1):
    loc_x = [i for i in range(-N, N+1)]
    loc_y = [i for i in range(-N, N+1)]

    vp.request.flush()
    s = time.time()
    for _ in range(I):
        dout_v = dout[:, N:-N, N:-N]
        dout_v[...] = 0
        for lx in loc_x:
            for ly in loc_y:
                if lx != 0 and ly != 0:
                    continue
                dout_v += din[:, N+ly:din.shape[-2]-N+ly, N+lx:din.shape[-1]-N+lx]
    vp.request.flush()
    e = time.time()
    return e - s


def nlcpy_sca_impl(din, dout, N, I=1):
    loc_x = [i for i in range(-N, N+1)]
    loc_y = [i for i in range(-N, N+1)]
    sin, sout = vp.sca.create_descriptor((din, dout))
    d = vp.sca.empty_description()
    for lx in loc_x:
        for ly in loc_y:
            if lx != 0 and ly != 0:
                continue
            d += sin[0, ly, lx]
    kern = vp.sca.create_kernel(d, sout[0, 0, 0])

    vp.request.flush()
    s = time.time()
    for _ in range(I):
        kern.execute()
    vp.request.flush()
    e = time.time()
    return e - s


def stencil_xya(din, dout, N, I=1, xp=np, lib=nb):
    if lib is nb:
        rt = numba_impl(din, dout, N, I)
    if lib is vp_naive:
        rt = nlcpy_naive_impl(din, dout, N, I)
    if lib is vp_sca:
        rt = nlcpy_sca_impl(din, dout, N, I)
    return rt
