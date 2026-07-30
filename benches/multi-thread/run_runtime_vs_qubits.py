import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm
import quantlop as ql


num_qubits = range(1, 25)
num_terms = 100
num_threads = (2, 4, 8, 16, 32, 64)

simulations = ["Serial"] + [f"{nt} threads" for nt in num_threads]
results = {method: {sim: {} for sim in simulations} for method in ("Higham", "Krylov")}

for nq in num_qubits:
    print(f"Simulation with {nq} qubits")
    psi = np.zeros(2**nq, dtype=complex)
    psi[0] = 1.0
    ham = ql.utils.get_rand_hamiltonian(nq, num_terms)

    start = time.perf_counter()
    serial_higham = ql.evolve_higham(ham, psi)
    end = time.perf_counter()
    label = f"{nq} qubits"
    results["Higham"]["Serial"][label] = end - start

    start = time.perf_counter()
    serial_krylov = ql.evolve_krylov(ham, psi)
    end = time.perf_counter()
    results["Krylov"]["Serial"][label] = end - start
    assert np.allclose(serial_higham, serial_krylov)

    for nt in tqdm(num_threads):
        start = time.perf_counter()
        higham = ql.evolve_higham(ham, psi, num_threads=nt)
        end = time.perf_counter()
        results["Higham"][f"{nt} threads"][label] = end - start

        start = time.perf_counter()
        krylov = ql.evolve_krylov(ham, psi, num_threads=nt)
        end = time.perf_counter()
        results["Krylov"][f"{nt} threads"][label] = end - start
        assert np.allclose(higham, krylov)

    output_path = Path(__file__).parent / "runtime_vs_qubits.json"
    with output_path.open(mode="w") as file:
        json.dump(results, file, indent=4)
