#include <bit>
#include <utility>

#include <quantlop/pauliword.hpp>

PauliWord::PauliWord(Complex coeff, String string)
    : coeff_(coeff),
      string_(std::move(string)),
      dimension_(Size(1) << string_.size())
{
    int y_count = 0;
    for (Index qubit = 0; qubit < string_.size(); ++qubit)
    {
        const Mask bit = Mask(1) << (string_.size() - 1 - qubit);
        switch (string_[qubit])
        {
        case 'X':
            flip_mask_ |= bit;
            break;
        case 'Y':
            flip_mask_ |= bit;
            phase_mask_ |= bit;
            ++y_count;
            break;
        case 'Z':
            phase_mask_ |= bit;
            break;
        default:
            break;
        }
    }

    const Complex imaginary_unit(0.0, 1.0);
    switch (y_count & 3)
    {
    case 0:
        base_phase_ = 1.0;
        break;
    case 1:
        base_phase_ = imaginary_unit;
        break;
    case 2:
        base_phase_ = -1.0;
        break;
    default:
        base_phase_ = -imaginary_unit;
        break;
    }
}

void PauliWord::apply(const Complex *in, Complex *out, int num_threads) const
{
    const int threads = num_threads > 0 ? num_threads : 1;

#pragma omp parallel for if (num_threads > 1) num_threads(threads) schedule(static)
    for (Index input_index = 0; input_index < dimension_; ++input_index)
    {
        const Mask output_index = Mask(input_index) ^ flip_mask_;
        const bool odd_parity = (std::popcount(phase_mask_ & Mask(input_index)) & 1) != 0;
        const Complex phase = odd_parity ? -base_phase_ : base_phase_;
        out[output_index] += coeff_ * phase * in[input_index];
    }
}

Size PauliWord::num_qubits() const { return string_.size(); }

Complex PauliWord::coeff() const { return coeff_; }

const String &PauliWord::string() const { return string_; }
