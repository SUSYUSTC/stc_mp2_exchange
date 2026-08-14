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
nsamples = get('nsamples')
mk = {'BN': 's', 'alkane': '^'}
fam_label = {'BN': '2D BN sheet', 'alkane': 'linear alkane'}
color = {'BN': config.color_STC, 'alkane': config.color_1}

ms = 9
fig, ax = plt.subplots(1, 1, figsize=(6, 4.8))


def plot(x, y, c, marker, label=None):
    o = np.argsort(x)
    x, y = x[o], y[o]
    plt.plot(x, y, 'k', linewidth=1.2, zorder=1)
    plt.scatter(x, y, s=ms**2, color=c, marker=marker, label=label, zorder=2,
                edgecolors='k', linewidths=0.6)


def powerfit(x, y, xfit):
    a, b = np.polyfit(np.log10(x), np.log10(y), 1)
    return a, 10**(a * np.log10(xfit) + b)


plt.sca(ax)
for fam in ['BN', 'alkane']:
    s = family == fam
    plot(nbf[s], nsamples[s], color[fam], mk[fam], fam_label[fam])
    # fit the asymptotic regime only (the smallest systems are pre-asymptotic)
    big = s & (nbf > 2000)
    xall = np.sort(nbf[s])
    a, yfit = powerfit(nbf[big], nsamples[big], xall)
    plt.plot(xall, yfit, '--', color=color[fam], linewidth=2, zorder=0,
             label=f'fit: $N^{{{a:.2f}}}$')
    print(f'nsamples {fam:8s} N^{a:.2f}')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of basis functions')
plt.title('$N_\\text{sample}$ for $\\epsilon = 0.3mE_h$')
plt.legend(frameon=False, loc='upper left', fontsize=13, **legend_opts)

libformat.set_log_ticks(ax.xaxis)
ax.set_xticks([200, 500, 1000, 2000, 5000])

plt.tight_layout()
plt.savefig("samples.png", dpi=200)
plt.savefig("samples.pdf")
plt.close()
