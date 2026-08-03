import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plot_settings import (  # noqa: E402
    FIGURE_STYLE,
    LINE_STYLE,
    PLOT_STYLE,
    save_plot,
)

DIRECTORY = Path(__file__).parent

PLOTS = {
    "runtime": "Runtime [s]",
    "memory": "Memory [MB]",
}

METHOD_STYLES = {
    "SciPy dense": {
        "label": "SciPy dense",
        "color": "#4C78A8",
        "marker": "o",
    },
    "SciPy sparse": {
        "label": "SciPy sparse",
        "color": "#B279A2",
        "marker": "s",
    },
    "Quantlop Higham": {
        "label": "Quantlop Higham",
        "color": "#54A24B",
        "marker": "^",
    },
    "Quantlop Krylov": {
        "label": "Quantlop Krylov",
        "color": "#F58518",
        "marker": "D",
    },
}


with mpl.rc_context(PLOT_STYLE):
    for filename, ylabel in PLOTS.items():
        with (DIRECTORY / f"{filename}.json").open() as file:
            results = json.load(file)

        fig, ax = plt.subplots(**FIGURE_STYLE)
        plotted_qubits = []

        for method, values in results.items():
            labels = sorted(values, key=int)
            qubits = np.array(labels, dtype=int)
            means = np.array([values[label] for label in labels], dtype=float).mean(axis=1)
            plotted_qubits.extend(qubits)
            ax.plot(qubits, means, **METHOD_STYLES[method], **LINE_STYLE)

        save_plot(fig, ax, plotted_qubits, ylabel, DIRECTORY / f"{filename}.svg")
