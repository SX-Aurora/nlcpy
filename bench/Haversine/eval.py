#
# * The source code in this file is based on the soure code of
#   weld-project/split-annotations.
#
# # NLCPy License #
#
#     Copyright (c) 2020 NEC Corporation
#     All rights reserved.
#
#     Redistribution and use in source and binary forms, with or without
#     modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
#     * Neither NEC Corporation nor the names of its contributors may be
#       used to endorse or promote products derived from this software
#       without specific prior written permission.
#
#     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#     ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#     WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#     DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#     FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#     (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#     LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#     ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#     (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#     SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# # weld-project/split-annotations License #
#
#    Redistribution and use in source and binary forms,
#    with or without modification, are permitted provided that the
#    following conditions are met:
#
#    1. Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the
#       distribution.
#
#    3. Neither the name of the copyright holder nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#    FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#    COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#    BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#    ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#    POSSIBILITY OF SUCH DAMAGE.
#

import numpy as np
import nlcpy as vp
from math import *
import time
import argparse
import csv
import ast
import json
import pandas as pd
import os
import string
import sys

# lat1, lon1, lat2, lon2

_test_source1_c = r'''
#include <stdint.h>
#include <stdlib.h>
#include <math.h>


void base_clock_(int *base) {
    char buf[256];
    if (ve_get_ve_info("clock_base", buf, sizeof(buf)) < 0)
    {
      *base = -1;
      return;
    }
    *base = atoi(buf);
    return;
}

void stm_(int64_t *val) {
    void *vehva = (void *)0x1000;
    __asm__ __volatile__ ("lhm.l %0,0(%1)":"=r"(*val):"r"(vehva));
    return;
}

float calc_harv(${dtype} lat1, ${dtype} lon1, ${dtype} *lat2, ${dtype} *lon2,
                int64_t num, ${dtype} *ans) {
    int ibase;
    int64_t ts, te;
    base_clock_(&ibase);
    stm_(&ts);

    ${dtype} miles_constant = 3959.0;
    ${dtype} dlat, dlon;
    ${dtype} a0, a1, a, c;

    lat1 = lat1 * M_PI / 180.0;
    lon1 = lon1 * M_PI / 180.0;

#ifdef FLOAT32
    #pragma omp parallel for
    for (int64_t i=0; i<num; i++){
        lat2[i] = lat2[i] * M_PI / 180.0;
        lon2[i] = lon2[i] * M_PI / 180.0;

        dlat = lat2[i] - lat1;
        dlon = lon2[i] - lon1;

        a0 = sinf(dlat/2.0);
        a1 = sinf(dlon/2.0);
        a = a0*a0 + cosf(lat1) * cosf(lat2[i]) * a1*a1;
        c = 2.0 * asinf(sqrtf(a));

        ans[i] = miles_constant * c;
    }
#else
#pragma omp parallel for
    for (int64_t i=0; i<num; i++){
        lat2[i] = lat2[i] * M_PI / 180.0;
        lon2[i] = lon2[i] * M_PI / 180.0;

        dlat = lat2[i] - lat1;
        dlon = lon2[i] - lon1;

        a0 = sin(dlat/2.0);
        a1 = sin(dlon/2.0);
        a = a0*a0 + cos(lat1) * cos(lat2[i]) * a1*a1;
        c = 2.0 * asin(sqrt(a));

        ans[i] = miles_constant * c;
    }
#endif

    stm_(&te);
    return (float)(te - ts) / ibase * 1e-6;
}
'''

# Read in the data
df = pd.read_csv('data', encoding='cp1252')
LATS_NAME_BASE = 'DATA/lats'
LONS_NAME_BASE = 'DATA/lons'

SIZE_NAME = 'result/size.dat'
T_NP_NAME = 'result/time_numpy.dat'
T_VP_NAME = 'result/time_nlcpy.dat'
T_J_NAME  = 'result/time_jit.dat'
T_JE_NAME  = 'result/time_jit_e.dat'

# calculation dtype np.float32 or np.float64
DT = np.float64

# vp.request.set_offload_timing_onthefly()

gentime = {np:0, vp:0, "nlcpy-jit":0}
intime  = {"pre":0, "exec(VE+VH)":0, "exec(VE)":0}
runtime = {np:0, vp:0, "nlcpy-jit":0}
exetime = {"nlcpy-jit":0}
result = {np:None, vp:None, "nlcpy-jit":None}

def _pytype2ctype(dtype):
    if dtype == np.float64:
        return "double"
    elif dtype == np.float32:
        return "float"
    else:
        raise TypeError


def use_nlcpy_jit_haversine(lat1, lon1, lat2, lon2, size):
    from nlcpy.ve_types import (uint32, uint64, int32, int64,
                                float32, float64, void_p, void)

    vp.request.flush()
    start = time.time()

    mod =  vp.jit.CustomVELibrary(
        code=string.Template(
            _test_source1_c
        ).substitute(
            dtype=_pytype2ctype(DT)
        ),
        compiler='/opt/nec/ve/bin/ncc',
        cflags=vp.jit.get_default_cflags(openmp=True) + \
                    (('-DFLOAT32',) if DT is np.float32 else ()),
        ldflags=vp.jit.get_default_ldflags(openmp=True),
        # ftrace=True,
    )

    if DT is np.float32:
        args_type = (float32, float32, uint64, uint64, int64, uint64)
    elif DT is np.float64:
        args_type = (float64, float64, uint64, uint64, int64, uint64)
    else:
        raise ValueError

    kern = mod.get_function(
        'calc_harv',
        args_type=args_type,
        ret_type=float32
    )

    end = time.time()
    intime["pre"] = end - start

    mi = vp.zeros(size, dtype=DT)
    vp.request.flush()
    start = time.time()
    ve_elapsed = kern(lat1, lon1, lat2.ve_adr, lon2.ve_adr, size, mi.ve_adr,
                      callback=None, sync=True)
    end = time.time()

    intime["exec(VE+VH)"] = end - start
    intime["exec(VE)"] = ve_elapsed

    return mi


# Haversine definition
def haversine(lat1, lon1, lat2, lon2, xp):
    miles_constant = 3959.0

    lat1 = lat1 * pi / 180.0
    lon1 = lon1 * pi / 180.0
    lat2 = lat2 * pi / 180.0
    lon2 = lon2 * pi / 180.0

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a0 = xp.sin(dlat/2.0)
    a1 = xp.sin(dlon/2.0)
    a = a0*a0 + xp.cos(lat1) * xp.cos(lat2) * a1*a1
    c = 2.0 * xp.arcsin(xp.sqrt(a))
    mi = miles_constant * c
    return mi



def write_to_file(arr, name):
    with open(name, 'a') as f:
        arr.tofile(f)


def gen_data(lat, lon, scale=10, xp=np):
    '''
    Generates the array replicated X times.
    '''

    if xp == "nlcpy-jit":
        xp = vp

    if xp is vp:
        vp.request.flush()

    start = time.time()
    new_lat = xp.arange(scale*len(lat),dtype=DT).reshape(scale, len(lat))
    new_lon = xp.arange(scale*len(lon),dtype=DT).reshape(scale, len(lon))
    new_lat += lat
    new_lon += lon
    new_lat = new_lat.ravel()
    new_lon = new_lon.ravel()
    if xp is vp:
        vp.request.flush()
    end = time.time()
    gen_time = end - start
    return new_lat, new_lon, gen_time


def compare(R1, R2):
    assert R1.dtype == R2.dtype, 'dtypes must match!'
    np.testing.assert_allclose(R1, R2, rtol=1e-4)


def print_args(args):
    d = vars(args)
    print('params: ', str(d))


def run_haversine(scale=10, use_numpy=True, use_nlcpy=False, use_nlcpy_jit=False):
    orig_lat = df['latitude'].values
    orig_lon = df['longitude'].values
    size = orig_lat.size * scale

    print(' dtype: ', DT.__name__)
    print('  size: ', size)

    use_liblist = []
    if use_nlcpy_jit:
        use_liblist.append("nlcpy-jit")
    if use_nlcpy:
        use_liblist.append(vp)
    if use_numpy:
        use_liblist.append(np)

    for xp in use_liblist:
        if xp is np:
            print("  numpy.......", end="", flush=True)
        elif xp is vp:
            print("  nlcpy.......", end="", flush=True)
        else:
            print("  nlcpy-jit...", end="", flush=True)

        lat, lon, gen_time = gen_data(orig_lat, orig_lon, scale=scale, xp=xp)
        gentime[xp] = gen_time

        if xp is vp:
            vp.request.flush()
        start = time.time()

        if xp == "nlcpy-jit" :
            dist = use_nlcpy_jit_haversine(40.671, -73.985, lat, lon, size)
        else:
            dist = haversine(40.671, -73.985, lat, lon, xp)

        if xp is vp:
            vp.request.flush()
        end = time.time()
        print("done")
        print("    generating data:", gen_time)
        runtime[xp] = end - start
        result[xp] = np.asarray(dist)
        print("    caluculation   :", runtime[xp])

        if xp == "nlcpy-jit":
             print("      ->  pre        : {:.17f}".format(intime["pre"]))
             print("      ->  exec(VE+VH): {:.17f}".format(intime["exec(VE+VH)"]))
             print("      ->  exec(VE)   : {:.17f}".format(intime["exec(VE)"]))
             print("      ->  other      : {:.17f}".format(runtime["nlcpy-jit"] -
                                                           intime["pre"] -
                                                           intime["exec(VE+VH)"]))

    if use_numpy & use_nlcpy & use_nlcpy_jit:
        compare(result[np], result[vp])
        compare(result[np], result['nlcpy-jit'])
    elif use_numpy & use_nlcpy:
        compare(result[np], result[vp])
    elif use_numpy & use_nlcpy_jit:
        compare(result[np], result['nlcpy-jit'])
    elif use_nlcpy & use_nlcpy_jit:
        compare(result[vp], result['nlcpy-jit'])

    # write result to file
    write_to_file(np.array(size, dtype='f8'), SIZE_NAME)
    if use_numpy:
        write_to_file(np.array(runtime[np], dtype='f8'), T_NP_NAME)
    if use_nlcpy:
        write_to_file(np.array(runtime[vp], dtype='f8'), T_VP_NAME)
    if use_nlcpy_jit:
        write_to_file(np.array(runtime["nlcpy-jit"], dtype='f8'), T_J_NAME)
        write_to_file(np.array(intime["exec(VE+VH)"], dtype='f8'), T_JE_NAME)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-scale', type=int, required=True,
                        help="size for scale up the orig dataset")
    parser.add_argument('--numpy', action='store_true',
                        help="use numpy or not in this run")
    parser.add_argument('--nlcpy', action='store_true',
                        help="use nlcpy or not in this run")
    parser.add_argument('--nlcpy-jit', action='store_true',
                        help="use jit compiler")

    args = parser.parse_args()

    arg = sys.argv
    print("arg:{}".format(arg))
    print_args(args)
    run_haversine(args.scale, args.numpy, args.nlcpy, args.nlcpy_jit)
