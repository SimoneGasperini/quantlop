#include <quantlop/pauliword.hpp>

namespace quantlop
{

PauliWord::PauliWord(Complex c, String str)
    : coeff(c),
      string(str),
      matvec(c, str)
{
}

PauliWord PauliWord::operator*(Complex c) const { return PauliWord(coeff * c, string); }

PauliWord operator*(Complex c, const PauliWord &pw) { return pw * c; }

Size PauliWord::num_qubits() const { return string.size(); }

Complex PauliWord::get_coeff() const { return coeff; }

const String &PauliWord::get_string() const { return string; }

}
