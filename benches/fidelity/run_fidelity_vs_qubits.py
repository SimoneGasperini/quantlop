import json
from pathlib import Path

import numpy as np
import scipy as sp
from tqdm import trange
import quantlop as ql


def compute_fidelity(psi1, psi2):
    return float(np.abs(np.vdot(psi1, psi2)) ** 2)


num_qubits = range(1, 15)
num_terms = 200
theta = 1.5
krylov_dims = (26, 28, 30)
num_reps = 50

methods = ["Higham"] + [f"Krylov-{dim}" for dim in krylov_dims]
results = {method: {} for method in methods}

for nq in num_qubits:
    label = f"{nq} qubits"
    for method in methods:
        results[method][label] = []

    for _ in trange(num_reps, desc=f"Run simulation {label}"):
        psi = np.zeros(2**nq, dtype=complex)
        psi[0] = 1.0
        ham = ql.utils.get_rand_hamiltonian(nq, num_terms, seed=42)
        exact = sp.linalg.expm(-1j * theta * ham.matrix()) @ psi

        evolved = ql.evolve_higham(ham, psi, theta)
        fid = compute_fidelity(exact, evolved)
        results["Higham"][label].append(fid)

        for dim in krylov_dims:
            evolved = ql.evolve_krylov(ham, psi, theta, dim_krylov=dim)
            fid = compute_fidelity(exact, evolved)
            results[f"Krylov-{dim}"][label].append(fid)

output_path = Path(__file__).parent / "fidelity_vs_qubits.json"
with output_path.open(mode="w") as file:
    json.dump(results, file, indent=4)
