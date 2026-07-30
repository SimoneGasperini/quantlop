#include <algorithm>
#include <utility>

#include <quantlop/hamiltonian.hpp>

Hamiltonian::Hamiltonian(std::vector<PauliWord> terms)
    : terms_(std::move(terms)),
      dimension_(terms_.front().dimension_)
{
    residual_indices_.reserve(terms_.size());
    for (Index index = 0; index < terms_.size(); ++index)
    {
        const PauliWord &term = terms_[index];
        if (term.flip_mask_ == 0 && term.phase_mask_ == 0)
        {
            identity_coeff_ += term.coeff_;
        }
        else
        {
            residual_indices_.push_back(index);
            residual_lcu_norm_ += std::abs(term.coeff_);
        }
    }
}

void Hamiltonian::residual_matvec_into(const Complex *in, Complex *out, int num_threads)
    const noexcept
{
    std::fill(out, out + dimension_, 0.0);
    for (const Index index : residual_indices_)
    {
        terms_[index].apply(in, out, num_threads);
    }
}

Size Hamiltonian::dimension() const noexcept { return dimension_; }

Size Hamiltonian::num_qubits() const noexcept { return terms_.front().num_qubits(); }

Size Hamiltonian::num_terms() const noexcept { return terms_.size(); }

double Hamiltonian::residual_lcu_norm() const noexcept { return residual_lcu_norm_; }

Complex Hamiltonian::identity_coeff() const noexcept { return identity_coeff_; }

const std::vector<PauliWord> &Hamiltonian::terms() const noexcept { return terms_; }
