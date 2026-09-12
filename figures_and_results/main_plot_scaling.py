import numpy as np
import config
import libformat
import matplotlib as mpl
fs = 16
mpl.rc('font', size=fs)
mpl.rc('axes', titlesize=fs)
import matplotlib.pyplot as plt
legend_opts = dict(handlelength=1.5, handletextpad=0.5, columnspacing=1.2)

names = open("./data_scaling").readline().split()
col = {n: i for i, n in enumerate(names)}
raw = [l.split() for l in open("./data_scaling").readlines()[1:]]
family = np.array([r[col['family']] for r in raw])


def get(key):
    return np.array([float('nan') if r[col[key]] == 'NA' else float(r[col[key]]) for r in raw])


nbf = get('nbf')
target = 0.3                       # requested stochastic error, mEh
mk = {'BN': 's', 'alkane': '^'}
fam_label = {'BN': '2D BN sheet', 'alkane': 'linear alkane'}

ms = 6
lw = 1.2
fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))


def plot(x, y, color, marker, label=None):
    o = np.argsort(x)
    x, y = x[o], y[o]
    good = np.isfinite(y)
    plt.plot(x[good], y[good], 'k', linewidth=lw, zorder=1)
    plt.scatter(x[good], y[good], s=ms**2, color=color, marker=marker, label=label,
                zorder=2, edgecolors='k', linewidths=0.6)


def powerfit(x, y):
    good = np.isfinite(y)
    return np.polyfit(np.log10(x[good]), np.log10(y[good]), 1)[0]


# ---------------------------------------------------- (a) BN anatomy: sets the colours
plt.sca(axes[0])
s = family == 'BN'
x = nbf[s]
t_scf = get('t_SCF_2it_incore')[s]
t_scf = np.where(np.isfinite(t_scf), t_scf, get('t_SCF_2it_outcore')[s]) / 2
series = [
    (t_scf, config.color_2, 'PySCF DF-HF (per iteration)'),
    (get('t_pyscf_MP2')[s], config.color_exact, 'PySCF DF-MP2'),
    (get('t_STC_laplace')[s] + get('t_STC_direct')[s], config.color_1, 'LT-MP2 direct'),
    (get('t_STC_exchange')[s], config.color_STC, 'STC-MP2 exchange'),
    (get('t_DLPNO_tight')[s], config.color_DLPNO, 'ORCA DLPNO-MP2'),
]
for y, color, label in series:
    plot(x, y, color, 's', label)
    print(f'{label:22s} N^{powerfit(x, y):.2f}')
plt.xscale('log')
plt.yscale('log')
# headroom so the legend clears the topmost curve
ymin, ymax = plt.ylim()
plt.ylim(ymin, ymax * 3)
plt.xlabel('Number of basis functions')
plt.ylabel('Wall time (s)')
plt.title('Timing of 2D BN sheets')
plt.legend(frameon=False, loc='upper left', fontsize=13, **legend_opts)

ms = 8

# ---------------------------------------------------- (b) timing: STC vs DLPNO
plt.sca(axes[1])
for fam in ['BN', 'alkane']:
    s = family == fam
    plot(nbf[s], get('t_STC_exchange')[s], config.color_STC, mk[fam])
    plot(nbf[s], get('t_DLPNO_tight')[s], config.color_DLPNO, mk[fam])
    print(f'STC exchange {fam:8s} N^{powerfit(nbf[s], get("t_STC_exchange")[s]):.2f}   '
          f'DLPNO N^{powerfit(nbf[s], get("t_DLPNO_tight")[s]):.2f}')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of basis functions')
plt.ylabel('Wall time (s)')
plt.title('Timing versus system type')

# ---------------------------------------------------- (c) error: STC vs DLPNO
plt.sca(axes[2])
for fam in ['BN', 'alkane']:
    s = family == fam
    plot(nbf[s], np.abs(get('err_DLPNO_tight')[s]) * 1000, config.color_DLPNO, mk[fam])
    print(f'DLPNO error {fam:8s} N^{powerfit(nbf[s], np.abs(get("err_DLPNO_tight")[s])):.2f}')
plt.axhline(target, color=config.color_STC, linewidth=3, zorder=2)
plt.xscale('log')
plt.yscale('log')
plt.ylim(3e-3, 1e2)
plt.xlabel('Number of basis functions')
plt.ylabel('Energy error (m$E_h$)')
plt.title('Accuracy versus system type')

for ax in axes:
    libformat.set_log_ticks(ax.xaxis)
    ax.set_xticks([200, 500, 1000, 2000, 5000])

# in (b, c) only the marker shape needs explaining; the colours are set in (a)
handles = [plt.Line2D([], [], color='0.75', marker=mk[f], linestyle='',
                      markeredgecolor='k', markeredgewidth=0.6, markersize=ms, label=fam_label[f])
           for f in ['BN', 'alkane']]
axes[1].legend(handles=handles, frameon=False, loc='upper left', fontsize=14, **legend_opts)
axes[2].legend(handles=handles, frameon=False, loc='lower right', fontsize=14, **legend_opts)
axes[2].text(0.97, 0.36, 'STC target error', transform=axes[2].transAxes, fontsize=14,
             color=config.color_STC, va='bottom', ha='right')

for ax, lab in zip(axes, 'abc'):
    ax.text(-0.15, 1.02, f'({lab})', transform=ax.transAxes, fontsize=20, fontweight="bold",
            va='bottom', ha='left')

plt.tight_layout()
plt.savefig("scaling.png", dpi=200)
plt.savefig("scaling.pdf")
plt.close()
