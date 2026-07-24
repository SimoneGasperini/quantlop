#pragma once

#include <quantlop/types.hpp>

class PauliWord
{
public:
    PauliWord(Complex coeff, String string);

    Size num_qubits() const;
    Complex coeff() const;
    const String &string() const;

private:
    friend class Hamiltonian;

    void apply(const Complex *in, Complex *out, int num_threads) const;

    Complex coeff_;
    String string_;
    Size dimension_;
    Mask flip_mask_ = 0;
    Mask phase_mask_ = 0;
    Complex base_phase_ = 1.0;
};
