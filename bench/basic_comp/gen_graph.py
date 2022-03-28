import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib
from matplotlib import pyplot as plt
from bench_op import modules

def set_additional_info(datasets):
    refs = datasets['numpy']['runtime']
    n = len(datasets['numpy']['runtime'])
    for m in modules:
        datasets[m]['speedup'] = [refs[i] / datasets[m]['runtime'][i] for i in range(n)]
        datasets[m]['module'] = [m for _ in range(n)]

# load picle data
path = 'result/{}_result.pickle'
datasets = {}
for m in modules:
    filepath = path.format(m)
    with open(filepath, mode='rb') as fi:
        datasets[m] = pickle.load(fi)
set_additional_info(datasets)

# create dataframe
dfs = []
for m in modules:
    dfs.append(pd.DataFrame(datasets[m]))
df = pd.concat(dfs)
df = df[df.module != 'numpy']
# df = df[df.array_size == '800MB']

print("df:\n", df)

sns.set(style='darkgrid', font_scale=1.2)

ax = sns.barplot(
    x='operations',
    y='speedup',
    hue='module',
    data=df,
    palette='Set1',
    ci=None
)
ax.set_xlabel('opetations', fontsize=12)
ax.set_ylabel('speedup', fontsize=14)
ax.set_title('Speedup ratio (NumPy=1), datasize=800MB, double precision', fontsize=18)
# ax.set_yscale('log')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid(which='minor', axis='y')
for p in ax.patches:
    ax.annotate(
        format(p.get_height(), '.2f'),
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha = 'center',
        va = 'top',
        xytext = (0, 10),
        textcoords = 'offset points',
        fontsize=10
    )

plt.xticks(rotation=60)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.savefig('result/basic_comp.png', dpi=600, bbox_inches='tight')
