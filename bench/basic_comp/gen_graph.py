import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib
from matplotlib import pyplot as plt

with open('result/basic_op_result.pickle', mode='rb') as fi:
    data = pickle.load(fi)

df = pd.DataFrame(data)

df_vp = df[df.module != 'numpy']
df_vp1 = df_vp[df_vp.array_size == '8MB']
df_vp2 = df_vp[df_vp.array_size == '800MB']

sns.set(style='darkgrid', font_scale=1.2)

ax = sns.barplot(
    x='operations',
    y='speedup',
    hue='array_size',
    data=df_vp,
    palette='Set1',
)
ax.set_xlabel('opetations', fontsize=12)
ax.set_ylabel('speedup', fontsize=14)
ax.set_title('NLCPy speedup ratio (NumPy=1)', fontsize=18)
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
