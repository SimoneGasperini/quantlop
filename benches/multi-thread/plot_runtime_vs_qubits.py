import json
from pathlib import Path

import numpy as np
import pylab as plt


colors = {
    "Serial": "black",
    "2 threads": "#3b528b",
    "4 threads": "#31678e",
    "8 threads": "#297a8e",
    "16 threads": "#218e8d",
    "32 threads": "#1fa188",
    "64 threads": "#2fb47c",
}

directory = Path(__file__).parent
filename = "runtime_vs_qubits"
with (directory / f"{filename}.json").open() as file:
    results = json.load(file)

for method, simulations in results.items():
    fig, ax = plt.subplots(figsize=(9, 6))

    for simulation, values in simulations.items():
        color = colors[simulation]
        labels = sorted(values.keys(), key=lambda label: int(label.split()[0]))
        qubits = [int(label.split()[0]) for label in labels]
        runtime = np.array([values[label] for label in labels], dtype=float)
        ax.plot(qubits, runtime, marker="o", color=color, label=simulation)

    ax.set_xlabel("Number of qubits", fontsize=14)
    ax.set_ylabel("Runtime [s]", fontsize=14)
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=12, ncols=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(directory / f"{method.lower()}_{filename}.pdf", bbox_inches="tight")
    plt.close(fig)
