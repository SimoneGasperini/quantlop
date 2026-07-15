#include <array>
#include <cmath>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <quantlop/hamiltonian.hpp>
#include <quantlop/pauliword.hpp>
#include <quantlop/simulation.hpp>
#include <quantlop/types.hpp>

namespace
{

bool close(quantlop::Complex lhs, quantlop::Complex rhs) { return std::abs(lhs - rhs) < 1e-12; }

void require(bool condition, const char *message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

}

int main()
{
    using quantlop::Complex;
    using quantlop::Hamiltonian;
    using quantlop::PauliWord;

    const Hamiltonian hamiltonian(std::vector<PauliWord>{PauliWord(1.0, "X")});
    const std::array<Complex, 2> initial{Complex(1.0, 0.0), Complex(0.0, 0.0)};

    std::array<Complex, 2> applied{};
    hamiltonian.matvec_into(initial.data(), applied.data());
    require(hamiltonian.num_qubits() == 1, "unexpected qubit count");
    require(close(applied[0], 0.0), "unexpected |0> matvec amplitude");
    require(close(applied[1], 1.0), "unexpected |1> matvec amplitude");

    const auto evolved =
        std::unique_ptr<Complex[]>(quantlop::evolve(hamiltonian, initial.data(), std::numbers::pi / 2.0, 0));
    require(close(evolved[0], 0.0), "unexpected |0> evolved amplitude");
    require(close(evolved[1], Complex(0.0, -1.0)), "unexpected |1> evolved amplitude");

    return 0;
}
