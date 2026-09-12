import numpy as np
import config
import libformat
import matplotlib as mpl
fs = 16
mpl.rc('font', size=fs)
mpl.rc('axes', titlesize=fs)
import matplotlib.pyplot as plt
legend_opts = dict(handlelength=1.5, handletextpad=0.5, columnspacing=1.2)

rows = [l.split() for l in open("./data_large") if not l.startswith('#')]
col = {n: i for i, n in enumerate(rows[0])}
raw = [r for r in rows[1:] if r[col['screen']] == '1.0e-02']


def get(key):
    return np.array([float(r[col[key]]) for r in raw])


label = [f"{r[col['system']]}/{r[col['basis']][-2].upper()}Z\n{r[col['nao']]} AOs" for r in raw]
order = np.argsort(get('nao'))

series = [(get('t_pyscf_MP2'), config.color_exact, 'PySCF DF-MP2'),
          (get('t_laplace') + get('t_direct'), config.color_1, 'LT-MP2 direct'),
          (get('t_exchange'), config.color_STC, 'STC-MP2 exchange')]

x = np.arange(len(raw))
width = 0.27


def make(scale, name):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for i, (y, color, lab) in enumerate(series):
        off = (i - 1) * width
        ax.bar(x + off, y[order], width, color=color, edgecolor='k', linewidth=0.8,
               label=lab, zorder=2)
        for xi, yi in zip(x + off, y[order]):
            pad = yi * 1.12 if scale == 'log' else yi + 0.015 * 7e4
            ax.text(xi, pad, f'{yi:.0f}', ha='center', va='bottom', fontsize=10)

    ax.set_yscale(scale)
    if scale == 'log':
        ax.set_ylim(40, 5e5)
        libformat.set_log_ticks(ax.yaxis)
        ax.set_yticks([1e2, 1e3, 1e4, 1e5])
    else:
        ax.set_ylim(0, 7.8e4)
        ax.set_yticks([0, 2e4, 4e4, 6e4])
        ax.set_yticklabels(['0', '20000', '40000', '60000'])
    ax.set_xticks(x)
    ax.set_xticklabels([label[i] for i in order], fontsize=12)
    ax.set_ylabel('Wall time (s)')
    ax.set_title('Benchmark on five realistic molecules')
    ax.legend(frameon=False, loc='upper left', ncols=3, fontsize=13, **legend_opts)

    plt.tight_layout()
    plt.savefig(f'{name}.png', dpi=200)
    plt.savefig(f'{name}.pdf')
    plt.close()


make('log', 'large')
make('linear', 'large_linear')

for r, e, d, s in zip(raw, get('t_pyscf_MP2'), get('t_laplace') + get('t_direct'), get('t_exchange')):
    print(f"{r[col['system']]:12s} {r[col['basis']]:8s} nao {r[col['nao']]:>5s}  "
          f"DF-MP2 {e:9.0f}  direct {d:8.0f}  exchange {s:7.0f}   "
          f"DF-MP2/exch {e/s:6.0f}x   direct/exch {d/s:5.1f}x")
