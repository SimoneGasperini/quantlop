#pragma once

#include <vector>

#include <quantlop/pauliword.hpp>
#include <quantlop/types.hpp>

class Hamiltonian
{
public:
    Hamiltonian(std::vector<PauliWord> terms);

    void residual_matvec_into(const Complex *in, Complex *out, int num_threads) const noexcept;

    Size dimension() const noexcept;
    Size num_qubits() const noexcept;
    Size num_terms() const noexcept;
    double residual_lcu_norm() const noexcept;
    Complex identity_coeff() const noexcept;

    const std::vector<PauliWord> &terms() const noexcept;

private:
    std::vector<PauliWord> terms_;
    std::vector<Index> residual_indices_;
    Size dimension_;
    double residual_lcu_norm_ = 0.0;
    Complex identity_coeff_ = 0.0;
};
