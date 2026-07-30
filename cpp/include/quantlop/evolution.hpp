#pragma once

#include <memory>

#include <quantlop/hamiltonian.hpp>
#include <quantlop/types.hpp>

std::unique_ptr<Complex[]> evolve_higham(
    const Hamiltonian &ham,
    const Complex *psi,
    double theta,
    double rtol,
    int num_threads);

std::unique_ptr<Complex[]> evolve_krylov(
    const Hamiltonian &ham,
    const Complex *psi,
    double theta,
    double rtol,
    int num_threads);
