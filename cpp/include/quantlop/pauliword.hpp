#pragma once
#include <quantlop/matvec.hpp>
#include <quantlop/types.hpp>

namespace quantlop
{

class PauliWord
{
public:
    PauliWord(Complex c, String str);
    PauliWord operator*(Complex c) const;
    friend PauliWord operator*(Complex c, const PauliWord &pw);

    Size num_qubits() const;

    Complex get_coeff() const;
    const String &get_string() const;

private:
    friend class Hamiltonian;

    Complex coeff;
    String string;
    MatVec matvec;
};

}
