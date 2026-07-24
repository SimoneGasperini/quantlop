#include <quantlop/simulation.hpp>

namespace quantlop
{

Complex *evolve(const Hamiltonian &ham, const Complex *psi, Complex coeff, int num_threads, int dim_krylov)
{
    return expm_multiply_krylov(ham, psi, coeff, num_threads, dim_krylov);
}

}
