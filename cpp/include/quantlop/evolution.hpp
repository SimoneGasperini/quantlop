#pragma once

#include <quantlop/hamiltonian.hpp>
#include <quantlop/types.hpp>

// clang-format off

Complex *evolve_higham(
    const Hamiltonian &ham,
    const Complex *psi,
    Complex theta,
    int num_threads);

Complex *evolve_krylov(
    const Hamiltonian &ham,
    const Complex *psi,
    Complex theta,
    int num_threads,
    int dim_krylov);
