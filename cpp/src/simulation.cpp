#include <quantlop/simulation.hpp>

namespace quantlop
{

Complex *evolve(const Hamiltonian &ham, const Complex *psi, Complex coeff)
{
    return expm_multiply_krylov(ham, psi, coeff);
}

} // namespace quantlop
