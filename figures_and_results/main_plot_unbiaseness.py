import numpy as np
import config
import matplotlib as mpl
fs = 20
mpl.rc('font', size=fs)
mpl.rc('axes', titlesize=fs)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
legend_opts = dict(handlelength=1.2, handletextpad=0.4, columnspacing=0.8)


def gauss(x):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-x**2 / 2)


E_mp2 = np.loadtxt("./data_unbiasedness", skiprows=3, usecols=1)
ref = -2.351443749949e+00
target_error = 3e-4

error = E_mp2 - ref
mean = error.mean()
std = error.std(ddof=1)
sem = std / np.sqrt(len(error))
print(f'n {len(error)}  mean {mean:.3e}  std {std:.3e}  sem {sem:.3e}  mean/sem {mean/sem:.2f}')

lw = 2.5
fig, ax1 = plt.subplots(1, 1, figsize=(6.5, 5))

plt.sca(ax1)
xlim = (-1.1, 1.1)
nbin = 20
bins = np.linspace(xlim[0], xlim[1], nbin + 1, endpoint=True)
plt.hist(error * 1000, bins=bins, color=config.color_STC)
ylim = (0, 400)
plt.plot([mean * 1000] * 2, ylim, 'k', label='mean', linestyle='--', linewidth=lw)
plt.fill_betweenx((0, 0), -sem * 1000, sem * 1000, color='gray', alpha=0.5, label='SEM')

x = np.linspace(xlim[0], xlim[1], 201, endpoint=True)
density = gauss(x / 1000 / target_error) * len(error) * (xlim[1] - xlim[0]) / nbin / (1000 * target_error)
plt.plot(x, density, color='k', label='$N(0, \\epsilon^2)$', linewidth=lw)

plt.xlim(xlim)
plt.ylim(ylim)
plt.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.02, 1), **legend_opts)
plt.xlabel('Energy error (m$E_h$)')
plt.ylabel('Distribution')
plt.title('2500 independent STC-MP2 runs')
plt.xticks([-1, -0.5, 0, 0.5, 1])
plt.yticks([])

# inset: the mean against its standard error, to show zero lies inside
ax1_ins = inset_axes(
    ax1,
    width="25%",
    height="25%",
    loc='upper left',
    bbox_to_anchor=(0.06, 0, 1, 1),
    bbox_transform=ax1.transAxes,
)

rect, line1, line2 = mark_inset(
    ax1, ax1_ins,
    loc1=1, loc2=3,      # which corners to connect
    fc="none",           # transparent box
    ec="0.4",            # edge color
    lw=1.5,
)
rect.set_linestyle('-')
for c in [line1, line2]:
    c.set_linestyle((0, (1, 1)))

plt.sca(ax1_ins)
plt.plot([mean * 1000] * 2, ylim, 'k', linestyle='--', linewidth=2)
plt.fill_betweenx((0, ylim[1] / 5), (mean - sem) * 1000, (mean + sem) * 1000,
                  color='gray', alpha=0.5, label='$\\pm \\frac{\\sigma}{\\sqrt{n}}$')
plt.ylim((0, ylim[1] / 10))
plt.yticks([])
plt.xlim(-0.02, 0.02)
plt.xticks([-0.02, 0, 0.02], ['-0.02', '0', '0.02'])

plt.tight_layout()
plt.savefig("unbiaseness.png", dpi=200)
plt.savefig("unbiaseness.pdf")
plt.close()
