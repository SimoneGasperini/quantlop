import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter


PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.labelweight": "medium",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
}

FIGURE_STYLE = {
    "figsize": (7.2, 4.8),
}

LINE_STYLE = {
    "linestyle": "-",
    "linewidth": 2.0,
    "markersize": 5.0,
    "markeredgecolor": "white",
    "markeredgewidth": 0.7,
    "solid_capstyle": "round",
    "zorder": 3,
}

LEGEND_STYLE = {
    "loc": "upper left",
    "ncols": 2,
    "frameon": True,
    "fancybox": False,
    "framealpha": 0.95,
    "facecolor": "white",
    "edgecolor": "#D5D9DD",
    "borderpad": 0.7,
    "columnspacing": 1.4,
    "handlelength": 2.4,
}

TIGHT_LAYOUT_PAD = 0.8
SAVEFIG_STYLE = {
    "bbox_inches": "tight",
}


def set_style(ax):
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="both", which="major", length=4, width=0.8)
    ax.tick_params(axis="both", which="minor", length=2.5, width=0.6)
    ax.grid(axis="y", which="major", color="#C8CDD2", linewidth=0.8, alpha=0.75)
    ax.grid(axis="y", which="minor", color="#E2E5E8", linewidth=0.5, alpha=0.55)
    ax.grid(axis="x", which="major", color="#E2E5E8", linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    ax.legend(**LEGEND_STYLE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_plot(fig, ax, qubits, ylabel, output_path):
    xlabel = "Number of qubits"
    xmin = min(qubits)
    xmax = max(qubits)
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(xmin - 0.5, xmax + 0.5), yscale="log")
    ax.set_xticks(np.arange(xmin, xmax + 1))
    set_style(ax)
    fig.tight_layout(pad=TIGHT_LAYOUT_PAD)
    fig.savefig(output_path, **SAVEFIG_STYLE)
    fig.savefig(output_path.with_suffix(".svg"), **SAVEFIG_STYLE)
    plt.close(fig)
