from __future__ import print_function
import numpy as np
import nlcpy as vp
import numba
from math import *
import time
import argparse
import csv
import ast
import json
import os
from compute_xa import stencil_xa
from compute_xya import stencil_xya
from compute_xyza import stencil_xyza

# calculation dtype
DT = np.float32

# target libraries
nb = 'numba'
vp_naive = 'nlcpy_naive'
vp_sca = 'nlcpy_sca'


#vp.request.set_offload_timing_onthefly()


def init_grid(NX, NY, NZ, dtype=np.float32, xp=np):
    rng = np.random.default_rng(0)
    grid = rng.random((NZ, NY, NX), dtype=dtype)
    return xp.asarray(grid)


def write_to_file(arr, name):
    with open(name, 'a') as f:
        arr.tofile(f)


def compare(n, v):
    assert n.dtype == v.dtype, 'dtypes mismatch!'
    assert np.allclose(n, v.get()),  'result mismatch!'


def print_args(args):
    d = vars(args)
    print('params: ', str(d))


def run_stencil(NX, NY, NZ, N, I, stencil_shape='xa', use_numba=True,
                use_nlcpy_naive=False, use_nlcpy_sca=False):
    use_liblist = []
    if use_numba:
        use_liblist.append(nb)
    if use_nlcpy_naive:
        use_liblist.append(vp_naive)
    if use_nlcpy_sca:
        use_liblist.append(vp_sca)

    runtime = {nb:0, vp_naive:0, vp_sca:0}
    gflops = {nb:0, vp_naive:0, vp_sca:0}
    result = {nb:None, vp_naive:None, vp_sca:None}
    for lib in use_liblist:
        if lib is nb:
            xp = np
        elif lib in (vp_naive, vp_sca):
            xp = vp
        else:
            raise RuntimeError

        din = init_grid(NX, NY, NZ, dtype=DT, xp=xp)
        dout = xp.zeros_like(din)

        if lib is nb:
            print("\nnumba(CPU)...", end="", flush=True)
        elif lib is vp_naive:
            print("\nnlcpy(naive)...", end="", flush=True)
        elif lib is vp_sca:
            print("\nnlcpy(sca)...", end="", flush=True)

        if stencil_shape == 'xa':
            n = 2
            rt = stencil_xa(din, dout, N, I=I, xp=xp, lib=lib)
        elif stencil_shape == 'xya':
            n = 4
            rt = stencil_xya(din, dout, N, I=I, xp=xp, lib=lib)
        elif stencil_shape == 'xyza':
            n = 6
            rt = stencil_xyza(din, dout, N, I=I, xp=xp, lib=lib)
        
        print("done", flush=True)
        runtime[lib] = rt
        gflops[lib] = din.size * (n*N+1) * I / runtime[lib] / 10**9
        result[lib] = dout
        print("  elapsed time:", runtime[lib]) 
        print("  GFLOPS:", gflops[lib]) 
    print()

    if use_numba and use_nlcpy_naive:
        compare(result[nb], result[vp_naive])
    if use_numba and use_nlcpy_sca:
        compare(result[nb], result[vp_sca])

    # write result to file 
    write_to_file(np.array((NX, NY, NZ), dtype='f8'),
        'result/size-{}{}.dat'.format(N, stencil_shape))
    if use_numba:
        write_to_file(np.array(gflops[nb], dtype='f8'),
        'result/{}-{}{}.dat'.format(nb, N, stencil_shape))
    if use_nlcpy_naive:
        write_to_file(np.array(gflops[vp_naive], dtype='f8'),
        'result/{}-{}{}.dat'.format(vp_naive, N, stencil_shape))
    if use_nlcpy_sca:
        write_to_file(np.array(gflops[vp_sca], dtype='f8'),
        'result/{}-{}{}.dat'.format(vp_sca, N, stencil_shape))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-nx', type=int, required=True,
                        help="number of x-axis grid points")
    parser.add_argument('-ny', type=int, required=True,
                        help="number of y-axis grid points")
    parser.add_argument('-nz', type=int, required=True,
                        help="number of z-axis grid points")
    parser.add_argument('-n', type=int, required=True,
                        help="number of relative index for x-axis")
    parser.add_argument('-it', type=int, required=True,
                        help="max iteration number")
    parser.add_argument('-stencil_shape', type=str, required=True,
                        help='computation stencil shape')
    parser.add_argument("--numba", action='store_true',
                        help="use numba or not in this run",
                        default=True)
    parser.add_argument("--nlcpy_naive", action='store_true',
                        help="use nlcpy naive or not in this run")
    parser.add_argument("--nlcpy_sca", action='store_true',
                        help="use nlcpy sca or not in this run")
    
    args = parser.parse_args()
    
    print_args(args)
    
    NX = args.nx
    NY = args.ny
    NZ = args.nz
    N = args.n
    I = args.it

    run_stencil(NX, NY, NZ, N, I, args.stencil_shape, args.numba, args.nlcpy_naive, args.nlcpy_sca)
