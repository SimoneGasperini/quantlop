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


COLORS = {
    "Serial": "black",
    "4 threads": "#31678e",
    "8 threads": "#297a8e",
    "16 threads": "#218e8d",
    "32 threads": "#1fa188",
    "64 threads": "#2fb47c",
}

DIRECTORY = Path(__file__).parent


with mpl.rc_context(PLOT_STYLE):
    with (DIRECTORY / "runtime.json").open() as file:
        results = json.load(file)

    for method, simulations in results.items():
        fig, ax = plt.subplots(**FIGURE_STYLE)
        plotted_qubits = []

        for sim, values in simulations.items():
            labels = sorted(values, key=int)
            qubits = np.array(labels, dtype=int)
            runtime = np.array([values[label] for label in labels], dtype=float)
            plotted_qubits.extend(qubits)
            ax.plot(qubits, runtime, marker="o", color=COLORS[sim], label=sim, **LINE_STYLE)

        save_plot(fig, ax, plotted_qubits, "Runtime [s]", DIRECTORY / f"{method.lower()}.pdf")
