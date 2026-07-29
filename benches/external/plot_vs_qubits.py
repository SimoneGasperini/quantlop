import json
from pathlib import Path

import numpy as np
import pylab as plt


colors = {
    "Scipy dense": "tab:blue",
    "Scipy sparse": "tab:purple",
    "Quantlop Higham": "tab:green",
    "Quantlop Krylov": "tab:orange",
}

directory = Path(__file__).parent
plots = {
    "runtime_vs_qubits": "Runtime [s]",
    "memory_vs_qubits": "Memory [MB]",
}

for filename, ylabel in plots.items():
    with (directory / f"{filename}.json").open() as file:
        results = json.load(file)

    fig, ax = plt.subplots(figsize=(9, 6))

    for method, values in results.items():
        color = colors[method]
        labels = sorted(values.keys(), key=int)
        qubits = [int(label) for label in labels]
        data = np.array([values[label] for label in labels], dtype=float)
        ax.plot(qubits, data, marker="o", color=color, label=method)

    ax.set_xlabel("Number of qubits", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(directory / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)
