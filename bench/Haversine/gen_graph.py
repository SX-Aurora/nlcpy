#
# * The source code in this file is developed independently by NEC Corporation.
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

import argparse
import subprocess
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

PNG_BAR_PATH = 'result/haversine_bar.png'
PNG_LINE_PATH = 'result/haversine_line.png'
SIZE_PATH = 'result/size.dat'
T_NP_PATH = 'result/time_numpy.dat'
T_VP_PATH = 'result/time_nlcpy.dat'
T_J_PATH  = 'result/time_jit.dat'
T_JE_PATH = 'result/time_jit_e.dat'

def gen_graph_bar(size, t_numpy, t_nlcpy, t_jit, t_jit_e):
    index = np.arange(len(t_numpy))
    labels = ["size={:,}".format(s) for s in size]

    fig, ax = plt.subplots()

    plt.rcParams["font.size"] = 12
    ax.xaxis.grid(ls="--")
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci',axis='x',scilimits=(0,0))

    bar_height = 0.2
    alpha = 0.8

    plt.title('haversine performance')
    plt.xlabel('size/elapsed_time $\propto$ FLOPS', fontsize=14)

    plt.barh(index + bar_height, t_nlcpy, bar_height,
    alpha=alpha, label='nlcpy', align='center')

    plt.barh(index, t_numpy, bar_height,
    alpha=alpha, label='numpy', align='center')

    plt.barh(index + bar_height*2, t_jit, bar_height,
    alpha=alpha, label='nlcpy-jit', align='center')

    plt.yticks(index + bar_height/2, labels)
    plt.tick_params(labelsize=14)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.savefig(PNG_BAR_PATH, bbox_inches='tight', dpi=600)


def gen_graph_line(size, t_numpy, t_nlcpy, t_jit, t_jit_e):
    fig = plt.figure()
    plt.rcParams["font.size"] = 12
    ax = fig.add_subplot(1,1,1)

    ax.plot(size, t_numpy, label='numpy', linestyle='--', marker='o')
    ax.plot(size, t_nlcpy, label='nlcpy', linestyle='--', marker='o')
    ax.plot(size, t_jit, label='nlcpy-jit(pre+exec)', linestyle='--', marker='x')
    ax.plot(size, t_jit_e, label='nlcpy-jit(exec)', linestyle='--', marker='x')

    ax.set_xlabel('size', fontsize=14)
    ax.set_ylabel('elapsed time[sec]', fontsize=14)
    ax.legend(loc='best', fontsize=14)
    ax.set_title('haversine performance', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.savefig(PNG_LINE_PATH, bbox_inches='tight', dpi=600)


if __name__ == "__main__":
    size = np.fromfile(SIZE_PATH).astype('i8')
    t_numpy = np.fromfile(T_NP_PATH)
    t_nlcpy = np.fromfile(T_VP_PATH)
    t_jit = np.fromfile(T_J_PATH)
    t_jit_e = np.fromfile(T_JE_PATH)

    # generate line graph
    gen_graph_line(size, t_numpy, t_nlcpy, t_jit, t_jit_e)

    # generate bar graph
    t_numpy = size/t_numpy
    t_nlcpy = size/t_nlcpy
    t_jit = size/t_jit
    t_jit_e = size/t_jit_e

    gen_graph_bar(size[::-1], t_numpy[::-1], t_nlcpy[::-1], t_jit[::-1], t_jit_e[::-1])
