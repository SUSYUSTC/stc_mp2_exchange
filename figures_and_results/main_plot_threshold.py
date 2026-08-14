import numpy as np
import config
import libformat
import matplotlib as mpl
fs = 16
mpl.rc('font', size=fs)
mpl.rc('axes', titlesize=fs)
import matplotlib.pyplot as plt
legend_opts = dict(handlelength=1.5, handletextpad=0.5, columnspacing=1.2)

# BN 4x4 sheet / cc-pVTZ, requested error 3e-4 Ha, screening threshold sweep
names = open("./data_threshold").readline().split()
d = np.loadtxt("./data_threshold", skiprows=1)
c = {n: d[:, i] for i, n in enumerate(names)}
screen = c['screen']
E_ref = 3.298692185907

ms = 9
lw = 1.2
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))


def plot(x, y, color, marker, label):
    plt.plot(x, y, 'k', linewidth=lw, zorder=1)
    plt.scatter(x, y, s=ms**2, color=color, marker=marker, label=label, zorder=2,
                edgecolors='k', linewidths=0.6)


# ------------------------------------------------- (a) size of the exact domain
plt.sca(axes[0])
plot(screen, c['nvir_per_occ_rms'], config.color_exact, 'o', None)
plt.xscale('log')
plt.ylim(0, 1150)
plt.xlabel('Screening threshold $\\tau$')
plt.ylabel('Virtuals per occupied (rms)')
plt.title('Size of the exact domain')

# ---------------------------------------------------------------- (b) timing
plt.sca(axes[1])
plot(screen, c['t_exact'], config.color_exact, 'o', 'deterministic')
plot(screen, c['t_sample'], config.color_2, '^', 'STC')
#plot(screen, c['t_exchange'], config.color_STC, 's', 'total')
imin = np.argmin(c['t_exchange'])
#plt.scatter(screen[imin], c['t_exchange'][imin], s=(ms * 2.6)**2, facecolors='none',
#            edgecolors=config.color_STC, linewidths=2.2, zorder=3)
plt.xscale('log')
plt.yscale('log')
plt.ylim(2e-2, 7.5e3)
plt.xlabel('Screening threshold $\\tau$')
plt.ylabel('Wall time (s)')
plt.title('Cost of each channel')

# ------------------------------------------------- (c) what the threshold discards
plt.sca(axes[2])
plot(screen, np.abs(c['E_exact'] - E_ref), config.color_exact, 'o', 'deterministic')
plt.axhline(3e-4, color='0.4', linestyle='--', linewidth=2, zorder=0)
plt.text(2.2e-1, 4.5e-4, 'STC target error', fontsize=13, color='0.4', ha='right', va='bottom')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-5, 3e1)
plt.xlabel('Screening threshold $\\tau$')
plt.ylabel('Energy error ($E_h$)')
plt.title('Deterministic truncation error')

for ax, lab in zip(axes, 'abc'):
    ax.text(-0.15, 1.02, f'({lab})', transform=ax.transAxes, fontsize=20, fontweight="bold",
            va='bottom', ha='left')
    libformat.set_log_ticks(ax.xaxis)
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1])
    ax.xaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext())

handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, loc='upper center', ncols=3,
           bbox_to_anchor=(0.5, 1.005), **legend_opts)
plt.tight_layout(rect=(0, 0, 1, 0.90))
plt.savefig("threshold.png", dpi=200)
plt.savefig("threshold.pdf")
plt.close()

print(f'optimum tau {screen[imin]:.0e}: t_exchange {c["t_exchange"][imin]:.2f} s, '
      f'{c["t_exchange"][0]/c["t_exchange"][imin]:.0f}x vs pure deterministic, '
      f'{c["t_exchange"][-1]/c["t_exchange"][imin]:.0f}x vs pure stochastic')
