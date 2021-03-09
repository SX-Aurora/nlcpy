import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib
import glob
import re
import pickle
from matplotlib import pyplot as plt

def make_graph_all():
    gb = glob.glob('result/*.pickle')
    for ax in gb:
        m = re.match('result/(\w+)-([0-9])D\.pickle', ax)
        name = m.group(1)
        num = m.group(2)
        dw_graph(name,num)

def dw_graph(name, num):
    with open('result/{}-{}D.pickle'.format(name, num), mode='rb') as fi:
        data = pickle.load(fi)

    df = pd.DataFrame(data)
    sns.set(style='darkgrid', font_scale=0.5)

    ax = sns.barplot(
        x='size', 
        y='speedup', 
        hue = 'module',
        data=df,
    )   

    ax.set_xlabel('number of array elements', fontsize=10)
    ax.set_ylabel('speedup', fontsize=12)
    ax.set_title('Speedup ratio [' + name + ',' + num + 'D]', fontsize=14)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(which='minor', axis='y')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)

    for p in ax.patches:
        ax.annotate(
            format(p.get_height(), '.4f'), 
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha = 'center', 
            va = 'top', 
            xytext = (0, 10), 
            textcoords = 'offset points',
            fontsize=6
        )

    plt.savefig('result/{}-{}D.png'.format(name, num),  dpi=600, bbox_inches='tight')
    ax.clear()
