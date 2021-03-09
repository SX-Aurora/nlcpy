import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib
from matplotlib import pyplot as plt

with open('result/ufunc_result.pickle', mode='rb') as fi:
    data = pickle.load(fi)

df = pd.DataFrame(data)

df_vp = df[df.module != 'numpy']
df_vp_bad = df_vp[df_vp['speedup'] < 1.0]

print("\n*** Bad Performance Operations Compared to NumPy\n")
print(df_vp_bad)
