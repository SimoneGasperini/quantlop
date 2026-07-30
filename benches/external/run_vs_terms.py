import json
import time
from pathlib import Path

import numpy as np
import quantlop as ql
import scipy as sp
from memory_profiler import memory_usage
from tqdm import tqdm

DIRECTORY = Path(__file__).parent
NUM_QUBITS = 12
NUM_TERMS = range(50, 401, 50)
NUM_REPS = 7
RNG_SEED = 5

METHODS = (
    "Scipy dense",
    "Scipy sparse",
    "Quantlop Higham",
    "Quantlop Krylov",
)


def scipy_dense(mat, psi):
    return sp.linalg.expm(-1j * mat) @ psi


def scipy_sparse(mat, psi):
    return sp.sparse.linalg.expm_multiply(-1j * mat, psi, traceA=0)


def quantlop_higham(ham, psi):
    return ql.evolve_higham(ham, psi)


def quantlop_krylov(ham, psi):
    return ql.evolve_krylov(ham, psi)


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

for num_terms in tqdm(NUM_TERMS, desc="Run simulation"):
    ham = ql.utils.get_rand_hamiltonian(
        num_qubits=NUM_QUBITS,
        num_terms=num_terms,
        rng=rng,
    )
    psi = np.zeros(2**NUM_QUBITS, dtype=complex)
    psi[0] = 1.0

    dense = ham.matrix()
    runtimes, memories, result_dense = run_repetitions(scipy_dense, dense, psi, num_reps=NUM_REPS)
    runtime_data["Scipy dense"][num_terms] = runtimes
    memory_data["Scipy dense"][num_terms] = memories
    del dense

    sparse = ham.sparse_matrix()
    runtimes, memories, result_sparse = run_repetitions(
        scipy_sparse, sparse, psi, num_reps=NUM_REPS
    )
    runtime_data["Scipy sparse"][num_terms] = runtimes
    memory_data["Scipy sparse"][num_terms] = memories
    del sparse

    runtimes, memories, result_higham = run_repetitions(
        quantlop_higham, ham, psi, num_reps=NUM_REPS
    )
    runtime_data["Quantlop Higham"][num_terms] = runtimes
    memory_data["Quantlop Higham"][num_terms] = memories

    runtimes, memories, result_krylov = run_repetitions(
        quantlop_krylov, ham, psi, num_reps=NUM_REPS
    )
    runtime_data["Quantlop Krylov"][num_terms] = runtimes
    memory_data["Quantlop Krylov"][num_terms] = memories

    assert np.allclose(result_dense, result_sparse)
    assert np.allclose(result_dense, result_higham)
    assert np.allclose(result_dense, result_krylov)

    save_results("runtime_vs_terms.json", runtime_data)
    save_results("memory_vs_terms.json", memory_data)
