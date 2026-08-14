import numpy as np
import matplotlib as mpl
mpl.rc('mathtext', fontset='cm')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from quimb import schematic


presets = {
    'bond': {'linewidth': 3},
    'phys': {'linewidth': 1.5},
    'center': {
        'color': '#75D2F0',
    },
    'A': {
        'color': (1.0, 0.4, 0.4),
    },
    'B': {
        'color': '#b8dbb3',
    },
    'C': {
        'color': '#fcb6a5',
    },
}

r = 0.42
text_kwargs = {'ha': 'center', 'va': 'center', 'fontsize': 22, 'color': 'k'}
index_kwargs = {'ha': 'center', 'va': 'center', 'fontsize': 22, 'color': 'k'}
cond_kwargs = {'ha': 'center', 'va': 'center', 'fontsize': 18, 'color': '#555555'}

l = 1.6
BL, BR = (-l, -l), (l, -l)
TL, TR = (-l, l), (l, l)
sep = 0.13          # half-separation of a double bond


def limits(ax, pad_left=0.0, pad_right=0.0, pad_bottom=0.0, pad_top=0.0):
    ax.set_aspect('equal', adjustable='box')
    ax.relim(); ax.autoscale_view()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    return (xmin - pad_left, xmax + pad_right, ymin - pad_bottom, ymax + pad_top)


def draw_J(ax):
    # J = sum_xy (sum_ia R_iax R_iay)(sum_jb R_jbx R_jby): the pairs (i,a) and (j,b)
    # are each shared by two tensors, so the diagram is a four-cycle with two double
    # bonds, which is why J factorizes.
    d = schematic.Drawing(presets=presets, ax=ax)
    d.line(BL, TL, preset='bond')                                      # x
    d.line(BR, TR, preset='bond')                                      # y
    for off in (-sep, sep):
        d.line((-l, -l + off), (l, -l + off), preset='bond')           # i and a
        d.line((-l, l + off), (l, l + off), preset='bond')             # j and b
    for p in [BL, BR, TL, TR]:
        d.circle(p, radius=r, preset='center')
        ax.text(*p, "$\\tilde{\\boldsymbol{R}}$", **text_kwargs)

    ax.text(-l - 0.3, 0, "$P$", **index_kwargs)
    ax.text(l + 0.3, 0, "$Q$", **index_kwargs)
    ax.text(0, -l - sep - 0.3, "$i$", **index_kwargs)
    ax.text(0, -l + sep + 0.3, "$a$", **index_kwargs)
    ax.text(0, l + sep + 0.3, "$j$", **index_kwargs)
    ax.text(0, l - sep - 0.3, "$b$", **index_kwargs)
    return limits(ax, 0.5, 0.5, 0.7, 0.7)


def draw_K(ax):
    # Every pair of the four tensors shares exactly one index: the complete graph.
    d = schematic.Drawing(presets=presets, ax=ax)
    for p, q in [(BL, TL), (BR, TR), (TL, TR), (BL, BR), (BL, TR), (TL, BR)]:
        d.line(p, q, preset='bond')
    for p in [BL, BR, TL, TR]:
        d.circle(p, radius=r, preset='center')
        ax.text(*p, "$\\tilde{\\boldsymbol{R}}$", **text_kwargs)

    ax.text(-l - 0.3, 0, "$P$", **index_kwargs)
    ax.text(l + 0.3, 0, "$Q$", **index_kwargs)
    ax.text(0, -l - 0.3, "$a$", **index_kwargs)
    ax.text(0, l + 0.3, "$b$", **index_kwargs)
    ax.text(0.55, 0.05, "$i$", **index_kwargs)
    ax.text(-0.55, 0.05, "$j$", **index_kwargs)
    return limits(ax, 0.5, 0.5, 0.45, 0.45)


def draw_guide(ax):
    # The two upper tensors are each split into an A and a C factor (dashed outlines);
    # the occupied pair is enumerated exactly, which leaves a tree.
    d = schematic.Drawing(presets=presets, ax=ax)
    s = 0.68
    A_L, C_L = (-l, l), (-l - s, l - s)
    A_R, C_R = (l, l), (l + s, l - s)

    d.line(C_L, BL, preset='bond')         # x
    d.line(C_R, BR, preset='bond')         # y
    d.line(A_L, A_R, preset='bond')        # b
    d.line(BL, BR, preset='bond')          # a
    d.line(A_R, BL, preset='bond')         # i
    d.line(A_L, BR, preset='bond')         # j

    for (A, C), angle in [((A_L, C_L), 45), ((A_R, C_R), -45)]:
        centre = (0.5 * (A[0] + C[0]), 0.5 * (A[1] + C[1]))
        span = np.hypot(A[0] - C[0], A[1] - C[1])
        ax.add_patch(Ellipse(centre, span + 2 * r + 0.24, 2 * r + 0.24, angle=angle,
                             fill=False, linestyle=(0, (4, 3)), linewidth=1.8,
                             edgecolor='0.35', zorder=5))

    for p in [BL, BR]:
        d.circle(p, radius=r, preset='center')
        ax.text(*p, "$|\\tilde{\\boldsymbol{R}}|$", **text_kwargs)
    for p in [A_L, A_R]:
        d.circle(p, radius=r, preset='B')
        ax.text(*p, "$\\boldsymbol{A}$", **text_kwargs)
    for p in [C_L, C_R]:
        d.circle(p, radius=r, preset='C')
        ax.text(*p, "$\\boldsymbol{B}$", **text_kwargs)

    ax.text(-l - 0.65, -0.15, "$P$", **index_kwargs)
    ax.text(l + 0.65, -0.15, "$Q$", **index_kwargs)
    ax.text(0, -l - 0.3, "$a$", **index_kwargs)
    ax.text(0, l + 0.3, "$b$", **index_kwargs)
    ax.text(0.55, 0.05, "$i$", **index_kwargs)
    ax.text(-0.55, 0.05, "$j$", **index_kwargs)

    ax.text(-l - 1.45, -0.15, "$p(P|ia)$", **cond_kwargs)
    ax.text(l + 1.45, -0.15, "$p(Q|ja)$", **cond_kwargs)
    ax.text(0, -l - 0.75, "$p(a|ij)$", **cond_kwargs)
    ax.text(0, l + 0.75, "$p(b|ij)$", **cond_kwargs)
    ax.text(0, -0.75, "$p(ij)$", **cond_kwargs)
    return limits(ax, 1.15, 1.15, 1.0, 1.0)


# ---------------------------------------------------------------- assemble
scale = 0.85            # inches per data unit
gap = 0.25              # inches between panels

fig = plt.figure()
axes = [fig.add_axes([0, 0, 1, 1]) for _ in range(3)]
lims = [draw_J(axes[0]), draw_K(axes[1]), draw_guide(axes[2])]

widths = [(x1 - x0) * scale for x0, x1, y0, y1 in lims]
heights = [(y1 - y0) * scale for x0, x1, y0, y1 in lims]
total_w = sum(widths) + 2 * gap
total_h = max(heights) + 0.45
fig.set_size_inches(total_w, total_h, forward=True)

titles = ['Direct term $J$', 'Exchange term $K$', 'STC loop-breaking of the exchange $K$']

left = 0.0
for ax, lim, w, h, lab, title in zip(axes, lims, widths, heights, 'abc', titles):
    ax.set_xlim(lim[0], lim[1])
    ax.set_ylim(lim[2], lim[3])
    ax.set_position([left / total_w, (total_h - h) / 2 / total_h, w / total_w, h / total_h])
    ax.set_axis_off()
    fig.text(left / total_w + 0.005, 0.99, f"({lab})", fontsize=20,
             fontweight='bold', va='top', ha='left')
    fig.text((left + 0.5 * w) / total_w, 0.99, title, fontsize=22,
             va='top', ha='center')
    left += w + gap

fig.savefig("diagram.png", dpi=200)
fig.savefig("diagram.pdf")
plt.close(fig)
