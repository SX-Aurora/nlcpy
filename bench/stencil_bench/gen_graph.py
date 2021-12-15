import params
import seaborn as sns
import pandas as pd
from numpy import testing
from matplotlib import pyplot as plt

def gen(dic, palettes, markers, name, save_csv=True):
    plt.clf()
    sns.set(style='darkgrid', font_scale=1.)
    df = pd.DataFrame(dic)
    if save_csv:
        df.to_csv(name + '.csv')
    print(df)
    check(df)
    sns.lineplot(
        x='size',
        y='GFLOPS',
        hue='target',
        style='target',
        markers=markers,
        dashes=False,
        palette=palettes,
        data=dic
    )
    plt.xticks(rotation=30)
    plt.savefig(name + '.png', bbox_inches='tight', dpi=600)

def check(df):
    df_targs = []
    for _targ in params.TARGS:
        df_targs.append(df.query("target == @_targ"))

    if len(df_targs) <= 1:
        return

    base = df_targs[0]
    for _df in df_targs[1:]:
        testing.assert_allclose(base['max'].values, _df['max'].values)
        testing.assert_allclose(base['min'].values, _df['min'].values)
        testing.assert_allclose(base['avg'].values, _df['avg'].values)
        testing.assert_allclose(base['std'].values, _df['std'].values)


if __name__ == '__main__':
    # 2D
    df = pd.read_csv('perf-xya.csv')
    gen(df, params.PALETTES, params.MARKERS, 'perf-xya', save_csv=False)
    # 3D
    df = pd.read_csv('perf-xyza.csv')
    gen(df, params.PALETTES, params.MARKERS, 'perf-xyza', save_csv=False)
