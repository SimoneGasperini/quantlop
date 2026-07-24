#pragma once
#include <quantlop/hamiltonian.hpp>
#include <quantlop/types.hpp>

namespace quantlop
{

Complex *expm_multiply_krylov(
    const Hamiltonian &ham,
    const Complex *psi,
    Complex coeff,
    int num_threads,
    int dim_krylov);

Complex *evolve(const Hamiltonian &ham, const Complex *psi, Complex coeff, int num_threads, int dim_krylov);

}
