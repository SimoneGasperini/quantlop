#include <quantlop/simulation.hpp>

namespace quantlop
{

Complex *evolve(const Hamiltonian &ham, const Complex *psi, Complex coeff, int num_threads)
{
    return expm_multiply_krylov(ham, psi, coeff, num_threads);
}

}
