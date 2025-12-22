import json
import numpy as np
import pylab as plt


with open("runtime.json", "r") as file:
    data = json.load(file)

label1, label2, label3 = data.keys()
color1, color2, color3 = "tab:blue", "tab:orange", "tab:green"
alpha = 0.3
yticks = [0.01, 0.1, 1, 10, 100]
qubits = sorted(int(i) for i in data[label1])
runtime1 = np.array([data[label1][str(q)] for q in qubits])
runtime2 = np.array([data[label2][str(q)] for q in qubits])
runtime3 = np.array([data[label3][str(q)] for q in qubits])

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(qubits, np.mean(runtime1, axis=1), marker="s", label=label1, color=color1)
for k in range(runtime1.shape[1]):
    ax.plot(qubits, runtime1[:, k], color=color1, alpha=alpha)
ax.plot(qubits, np.mean(runtime2, axis=1), marker="s", label=label2, color=color2)
for k in range(runtime2.shape[1]):
    ax.plot(qubits, runtime2[:, k], color=color2, alpha=alpha)
ax.plot(qubits, np.mean(runtime3, axis=1), marker="s", label=label3, color=color3)
for k in range(runtime3.shape[1]):
    ax.plot(qubits, runtime3[:, k], color=color3, alpha=alpha)
ax.set_xlabel("Qubits", fontsize=18)
ax.set_xticks(qubits)
ax.set_xticklabels(qubits, fontsize=14)
ax.set_ylabel("Runtime [s]", fontsize=18)
ax.set_yticks(yticks)
ax.set_yticklabels(yticks, fontsize=14)
ax.set_yscale("log")
ax.legend(fontsize=18)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid()
fig.savefig("runtime.pdf")
plt.show()


with open("memory.json", "r") as file:
    data = json.load(file)

yticks = [0.1, 1, 10, 100, 1000]
memory1 = np.array([data[label1][str(q)] for q in qubits])
memory2 = np.array([data[label2][str(q)] for q in qubits])
memory3 = np.array([data[label3][str(q)] for q in qubits])

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(qubits, np.mean(memory1, axis=1), marker="s", label=label1, color=color1)
for k in range(memory1.shape[1]):
    ax.plot(qubits, memory1[:, k], color=color1, alpha=alpha)
ax.plot(qubits, np.mean(memory2, axis=1), marker="s", label=label2, color=color2)
for k in range(memory2.shape[1]):
    ax.plot(qubits, memory2[:, k], color=color2, alpha=alpha)
ax.plot(qubits, np.mean(memory3, axis=1), marker="s", label=label3, color=color3)
for k in range(memory3.shape[1]):
    ax.plot(qubits, memory3[:, k], color=color3, alpha=alpha)
ax.set_xlabel("Qubits", fontsize=18)
ax.set_xticks(qubits)
ax.set_xticklabels(qubits, fontsize=14)
ax.set_ylabel("Memory [MB]", fontsize=18)
ax.set_yticks(yticks)
ax.set_yticklabels(yticks, fontsize=14)
ax.set_yscale("log")
ax.legend(fontsize=18)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid()
fig.savefig("memory.pdf")
plt.show()
