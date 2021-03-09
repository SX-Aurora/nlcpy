#
# * The source code in this file is based on the soure code of
#   weld-project/split-annotations.
#
# # NLCPy License #
#
#     Copyright (c) 2020-2021 NEC Corporation
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

# Read in the data
df = pd.read_csv('data', encoding='cp1252')
LATS_NAME_BASE = 'DATA/lats'
LONS_NAME_BASE = 'DATA/lons'

SIZE_NAME = 'result/size.dat'
T_NP_NAME = 'result/time_numpy.dat'
T_VP_NAME = 'result/time_nlcpy.dat'

# calculation dtype
DT = np.float64

# vp.request.set_offload_timing_onthefly()


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
    R2 = R2.get() # convert to numpy.ndarray
    assert R1.dtype == R2.dtype, 'dtypes must match!'
    np.testing.assert_allclose(R1, R2, rtol=1e-4)


def print_args(args):
    d = vars(args)
    print('params: ', str(d))


def run_haversine(scale=10, use_numpy=True, use_nlcpy=False):
    orig_lat = df['latitude'].values
    orig_lon = df['longitude'].values
    size = orig_lat.size * scale

    print("  size = ", size)

    use_liblist = []
    if use_numpy:
        use_liblist.append(np)
    if use_nlcpy:
        use_liblist.append(vp)

    runtime = {np:0, vp:0}
    result = {np:None, vp:None}
    for xp in use_liblist:
        if xp is np:
            print("  numpy...", end="", flush=True)
        else:
            print("  nlcpy...", end="", flush=True)

        lat, lon, gen_time = gen_data(orig_lat, orig_lon, scale=scale, xp=xp)
        if xp is vp:
            vp.request.flush()
            #vp.start_profiling()
        start = time.time()

        dist = haversine(40.671, -73.985, lat, lon, xp)

        if xp is vp:
            vp.request.flush()
            #vp.stop_profiling()
            #vp.print_run_stats()
        end = time.time()
        print("done", flush=True)
        print("    generating data:", gen_time)
        runtime[xp] = end - start
        result[xp] = dist
        print("    caluculation   :", runtime[xp])

    if use_numpy and use_nlcpy:
        compare(result[np], result[vp])

    # write result to file
    write_to_file(np.array(size, dtype='f8'), SIZE_NAME)
    if use_numpy:
        write_to_file(np.array(runtime[np], dtype='f8'), T_NP_NAME)
    if use_nlcpy:
        write_to_file(np.array(runtime[vp], dtype='f8'), T_VP_NAME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-scale', type=int, required=True,
                        help="size for scale up the orig dataset")
    parser.add_argument('--numpy', action='store_true',
                        help="use numpy or not in this run")
    parser.add_argument('--nlcpy', action='store_true',
                        help="use nlcpy or not in this run")

    args = parser.parse_args()

    print_args(args)

    run_haversine(args.scale, args.numpy, args.nlcpy)
