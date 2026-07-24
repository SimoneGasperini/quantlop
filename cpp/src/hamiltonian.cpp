#include <algorithm>
#include <utility>

#include <quantlop/hamiltonian.hpp>

Hamiltonian::Hamiltonian(std::vector<PauliWord> terms)
    : terms_(std::move(terms))
{
}

void Hamiltonian::matvec_into(const Complex *in, Complex *out, int num_threads) const
{
    const Size dimension = Size(1) << num_qubits();
    std::fill(out, out + dimension, 0.0);
    for (const PauliWord &term : terms_)
    {
        term.apply(in, out, num_threads);
    }
}

double Hamiltonian::lcu_norm() const
{
    double norm = 0.0;
    for (const PauliWord &term : terms_)
    {
        norm += std::abs(term.coeff_);
    }
    return norm;
}

Size Hamiltonian::num_qubits() const { return terms_.front().num_qubits(); }

Size Hamiltonian::num_terms() const { return terms_.size(); }

const std::vector<PauliWord> &Hamiltonian::terms() const { return terms_; }
