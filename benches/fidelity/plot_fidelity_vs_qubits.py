import json
from pathlib import Path

import numpy as np
import pylab as plt

colors = {
    "Higham": "black",
    "Krylov-26": "#6baed6",
    "Krylov-28": "#4292c6",
    "Krylov-30": "#2171b5",
}

directory = Path(__file__).parent
filename = "fidelity_vs_qubits"
with (directory / f"{filename}.json").open() as file:
    results = json.load(file)

fig, ax = plt.subplots(figsize=(9, 6))

for method, values in results.items():
    color = colors[method]
    labels = sorted(values.keys(), key=lambda label: int(label.split()[0]))
    qubits = [int(label.split()[0]) for label in labels]
    data = np.array([values[label] for label in labels], dtype=float)
    mean = data.mean(axis=1)
    minimum = data.min(axis=1)
    maximum = data.max(axis=1)

    ax.plot(qubits, mean, marker="o", color=color, label=method)
    ax.fill_between(qubits, minimum, maximum, color=color, alpha=0.2)

ax.set_xlabel("Number of qubits", fontsize=14)
ax.set_ylabel("Fidelity", fontsize=14)
ax.grid(alpha=0.3)
ax.legend(fontsize=14)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(directory / f"{filename}.pdf", bbox_inches="tight")
plt.close(fig)
