#pragma once
#include <vector>

#include <quantlop/pauliword.hpp>
#include <quantlop/types.hpp>

namespace quantlop
{

class Hamiltonian
{
public:
    Hamiltonian(std::vector<PauliWord> pws);
    void matvec_into(const Complex *in, Complex *out) const;
    void matvec_into(const Complex *in, Complex *out, int num_threads) const;
    Hamiltonian operator*(Complex c) const;
    friend Hamiltonian operator*(Complex c, const Hamiltonian &ham);
    double lcu_norm() const;

    Size num_qubits() const;
    Size num_terms() const;

    const std::vector<PauliWord> &get_pwords() const;

private:
    std::vector<PauliWord> pwords;
};

}
