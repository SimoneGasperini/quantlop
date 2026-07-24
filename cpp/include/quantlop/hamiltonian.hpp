#pragma once
#include <vector>

#include <quantlop/pauliword.hpp>
#include <quantlop/types.hpp>

class Hamiltonian
{
public:
    Hamiltonian(std::vector<PauliWord> terms);

    void matvec_into(const Complex *in, Complex *out, int num_threads) const;
    double lcu_norm() const;

    Size num_qubits() const;
    Size num_terms() const;

    const std::vector<PauliWord> &terms() const;

private:
    std::vector<PauliWord> terms_;
};
