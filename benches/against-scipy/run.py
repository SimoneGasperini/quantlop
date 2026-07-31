import json
import time
from pathlib import Path

import numpy as np
import quantlop as ql
import scipy as sp
from memory_profiler import memory_usage
from tqdm import tqdm

DIRECTORY = Path(__file__).parent
NUM_QUBITS = range(1, 24)
NUM_TERMS = 200
NUM_REPS = 7
RNG_SEED = 5

METHODS = (
    "SciPy dense",
    "SciPy sparse",
    "Quantlop Higham",
    "Quantlop Krylov",
)


def scipy_dense(mat, psi):
    return sp.linalg.expm(-1j * mat) @ psi


def scipy_sparse(mat, psi):
    return sp.sparse.linalg.expm_multiply(-1j * mat, psi, traceA=0)


def runtime_and_memory(func, *args, interval=0.0005):
    start = time.perf_counter()
    memory_samples, result = memory_usage(
        (func, args, {}),
        interval=interval,
        retval=True,
        include_children=True,
        multiprocess=True,
        max_iterations=1,
    )
    runtime = time.perf_counter() - start
    memory = max(memory_samples) - memory_samples[0]
    return runtime, memory, result


def run_repetitions(func, *args, num_reps):
    runtimes = []
    memories = []
    result = None

    for _ in range(num_reps):
        runtime, memory, result = runtime_and_memory(func, *args)
        runtimes.append(runtime)
        memories.append(memory)

    return runtimes, memories, result


def save_results(filename, results):
    with (DIRECTORY / filename).open("w") as file:
        json.dump(results, file, indent=4)


runtime_data = {method: {} for method in METHODS}
memory_data = {method: {} for method in METHODS}
rng = np.random.default_rng(seed=RNG_SEED)

for num_qubits in tqdm(NUM_QUBITS, desc="Run simulation"):
    ham = ql.utils.get_rand_hamiltonian(num_qubits, num_terms=NUM_TERMS, rng=rng)
    psi = np.zeros(2**num_qubits, dtype=complex)
    psi[0] = 1.0

    def run(method, func, *args):
        runtimes, memories, result = run_repetitions(func, *args, num_reps=NUM_REPS)
        runtime_data[method][num_qubits] = runtimes
        memory_data[method][num_qubits] = memories
        return result

    if num_qubits < 15:
        result_dense = run("SciPy dense", scipy_dense, ham.matrix(), psi)

    if num_qubits < 21:
        result_sparse = run("SciPy sparse", scipy_sparse, ham.sparse_matrix(), psi)

    result_higham = run("Quantlop Higham", ql.evolve_higham, ham, psi)
    result_krylov = run("Quantlop Krylov", ql.evolve_krylov, ham, psi)

    if num_qubits < 15:
        assert np.allclose(result_sparse, result_dense)
    if num_qubits < 21:
        assert np.allclose(result_sparse, result_higham)
        assert np.allclose(result_sparse, result_krylov)
    assert np.allclose(result_higham, result_krylov)

    save_results("runtime.json", runtime_data)
    save_results("memory.json", memory_data)
