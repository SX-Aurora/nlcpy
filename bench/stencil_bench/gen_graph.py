import argparse
import subprocess
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import seaborn as sns


def gen_graph_bar(n, nx, ny, nz, gf_nb, gf_vp_naive, gf_vp_sca, label):
    index = np.arange(len(n))
    if label is 'xa':
        labels = ["{}xa".format(_n) for _n in n]
    elif label is 'xya':
        labels = ["{}x{}ya".format(_n, _n) for _n in n]
    elif label is 'xyza':
        labels = ["{}x{}y{}za".format(_n, _n, _n) for _n in n]
    else:
        raise NotImplementedError

    fig, ax = plt.subplots()
    
    plt.rcParams["font.size"] = 12
    ax.xaxis.grid(ls="--")
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci',axis='x',scilimits=(0,0))

    bar_height = 0.25
    alpha = 0.8
    
    plt.title(
        'Single Precision (NX={}, NY={}, NZ={})'
        .format(int(nx[0]), int(ny[0]), int(nz[0])))
    plt.xlabel('GFLOPS', fontsize=14)
    
    plt.barh(index + bar_height * 2, gf_nb, bar_height,
    alpha=alpha, label='Numba(CPU)', align='center', color='dodgerblue')
    
    plt.barh(index + bar_height * 1, gf_vp_naive, bar_height,
    alpha=alpha, label='NLCPy(naive)', align='center', color='pink')

    plt.barh(index + bar_height * 0, gf_vp_sca, bar_height,
    alpha=alpha, label='NLCPy(SCA)', align='center', color='red')

    plt.yticks(index + bar_height/2, labels)
    plt.tick_params(labelsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.savefig('result/perf-{}.png'.format(label), bbox_inches='tight', dpi=600)



if __name__ == "__main__":
    labels = ['xa', 'xya', 'xyza']
    for l in labels:
        n = []
        nx = []
        ny = []
        nz = []
        gf_nb = []
        gf_vp_naive = []
        gf_vp_sca = []
        for i in range(1, 5):
            n.append(i)

            size = np.fromfile(
                'result/size-{}{}.dat'.format(i, l))
            nx.append(size[0])
            ny.append(size[1])
            nz.append(size[2])
            
            gf_nb_tmp = np.fromfile(
                'result/numba-{}{}.dat'.format(i, l))
            gf_nb.append(gf_nb_tmp[0])
            
            gf_vp_naive_tmp = np.fromfile(
                'result/nlcpy_naive-{}{}.dat'.format(i, l))
            gf_vp_naive.append(gf_vp_naive_tmp[0])
            
            gf_vp_sca_tmp = np.fromfile(
                'result/nlcpy_sca-{}{}.dat'.format(i, l))
            gf_vp_sca.append(gf_vp_sca_tmp[0])
        print("label:", l)
        print("n:", n)
        print("nx:", nx)
        print("ny:", ny)
        print("nz:", nz)
        print("gf_nb:", gf_nb)
        print("gf_vp_naive:", gf_vp_naive)
        print("gf_vp_sca:", gf_vp_sca)
        print()
        gen_graph_bar(
            n[::-1], nx[::-1], ny[::-1], nz[::-1],
            gf_nb[::-1], gf_vp_naive[::-1], gf_vp_sca[::-1], l)
