#pragma once
#include <quantlop/hamiltonian.hpp>
#include <quantlop/types.hpp>

namespace quantlop
{

Complex *expm_multiply_krylov(const Hamiltonian &ham, const Complex *psi, Complex coeff);
Complex *evolve(const Hamiltonian &ham, const Complex *psi, Complex coeff);

} // namespace quantlop
