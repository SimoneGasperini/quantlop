import json
import time
from pathlib import Path

import numpy as np
import quantlop as ql
from tqdm import tqdm


DIRECTORY = Path(__file__).parent
NUM_QUBITS = range(1, 26)
NUM_TERMS = 200
NUM_THREADS = (4, 8, 16, 32, 64)
RNG_SEED = 5
METHODS = {
    "Higham": ql.evolve_higham,
    "Krylov": ql.evolve_krylov,
}

simulations = ["Serial", *(f"{nt} threads" for nt in NUM_THREADS)]
results = {method: {sim: {} for sim in simulations} for method in METHODS}
rng = np.random.default_rng(seed=RNG_SEED)

for num_qubits in tqdm(NUM_QUBITS, desc="Run simulation"):
    psi = np.zeros(2**num_qubits, dtype=complex)
    psi[0] = 1.0
    ham = ql.utils.get_rand_hamiltonian(num_qubits, num_terms=NUM_TERMS, rng=rng)

    serial = {}
    for method, evolve in METHODS.items():
        start = time.perf_counter()
        serial[method] = evolve(ham, psi)
        results[method]["Serial"][num_qubits] = time.perf_counter() - start
    assert np.allclose(serial["Higham"], serial["Krylov"])

    for num_threads in tqdm(NUM_THREADS, desc=f"{num_qubits} qubits", leave=False):
        threaded = {}
        simulation = f"{num_threads} threads"
        for method, evolve in METHODS.items():
            start = time.perf_counter()
            threaded[method] = evolve(ham, psi, num_threads=num_threads)
            results[method][simulation][num_qubits] = time.perf_counter() - start
            assert np.allclose(serial[method], threaded[method])
        assert np.allclose(threaded["Higham"], threaded["Krylov"])

    with (DIRECTORY / "runtime.json").open("w") as file:
        json.dump(results, file, indent=4)
