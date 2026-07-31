#include <array>
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
        case 'I':
            break;
        }
    }

    static constexpr std::array<Complex, 4> y_phases =
        {Complex(1.0, 0.0), Complex(0.0, 1.0), Complex(-1.0, 0.0), Complex(0.0, -1.0)};
    factor_ = coeff_ * y_phases[y_count & 3];
}

void PauliWord::apply(const Complex *in, Complex *out, int num_threads) const noexcept
{
    const std::int64_t dimension = static_cast<std::int64_t>(dimension_);
    const int nt = num_threads > 0 ? num_threads : 1;

#pragma omp parallel for if (nt > 0) num_threads(nt) schedule(static)
    for (std::int64_t input_index = 0; input_index < dimension; ++input_index)
    {
        const Mask output_index = Mask(input_index) ^ flip_mask_;
        const bool odd_parity = (std::popcount(phase_mask_ & Mask(input_index)) & 1) != 0;
        const Complex factor = odd_parity ? -factor_ : factor_;
        out[output_index] += factor * in[input_index];
    }
}

Size PauliWord::num_qubits() const noexcept { return string_.size(); }

Complex PauliWord::coeff() const noexcept { return coeff_; }

const String &PauliWord::string() const noexcept { return string_; }
