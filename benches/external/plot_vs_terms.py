import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

from settings import (
    DIRECTORY,
    FIGURE_STYLE,
    LINE_STYLE,
    METHOD_STYLES,
    PLOT_LABELS,
    PLOT_STYLE,
    SAVEFIG_STYLE,
    TIGHT_LAYOUT_PAD,
    apply_common_axes_style,
)

NUM_QUBITS = 12
PLOTS = {
    "runtime_vs_terms": PLOT_LABELS["runtime"],
    "memory_vs_terms": PLOT_LABELS["memory"],
}

with mpl.rc_context(PLOT_STYLE):
    for filename, ylabel in PLOTS.items():
        with (DIRECTORY / f"{filename}.json").open() as file:
            results = json.load(file)

        fig, ax = plt.subplots(**FIGURE_STYLE)
        term_counts = sorted({int(label) for values in results.values() for label in values})

        for method, values in results.items():
            labels = sorted(values, key=int)
            terms = np.array([int(label) for label in labels])
            means = np.array([values[label] for label in labels], dtype=float).mean(axis=1)
            ax.plot(
                terms,
                means,
                **METHOD_STYLES[method],
                **LINE_STYLE,
            )

        ax.set(
            xlabel="Number of terms",
            ylabel=ylabel,
            xscale="log",
            yscale="log",
        )
        ax.set_xticks(term_counts)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        apply_common_axes_style(ax)
        ax.text(
            0.98,
            0.04,
            f"{NUM_QUBITS} qubits",
            transform=ax.transAxes,
            ha="right",
            color="#555555",
        )
        fig.tight_layout(pad=TIGHT_LAYOUT_PAD)
        fig.savefig(DIRECTORY / f"{filename}.pdf", **SAVEFIG_STYLE)
        plt.close(fig)
