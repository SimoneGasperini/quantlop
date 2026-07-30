#include <memory>
#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <quantlop/evolution.hpp>
#include <quantlop/hamiltonian.hpp>
#include <quantlop/pauliword.hpp>
#include <quantlop/types.hpp>

using namespace nanobind;

using ComplexArray = ndarray<const Complex, ndim<1>, c_contig, device::cpu>;
using NumpyComplexArray = ndarray<numpy, Complex, ndim<1>, c_contig>;

static NumpyComplexArray wrap_evolved_state(std::unique_ptr<Complex[]> out, Size dimension)
{
    Complex *out_ptr = out.release();
    capsule owner(out_ptr, [](void *p) noexcept { delete[] static_cast<Complex *>(p); });
    return NumpyComplexArray(out_ptr, {dimension}, owner);
}

template <auto evolve>
static NumpyComplexArray evolve_py(
    const Hamiltonian &ham,
    ComplexArray psi,
    double theta,
    double rtol,
    int num_threads)
{
    const Size dimension = psi.shape(0);
    std::unique_ptr<Complex[]> out;
    {
        gil_scoped_release release;
        out = evolve(ham, psi.data(), theta, rtol, num_threads);
    }
    return wrap_evolved_state(std::move(out), dimension);
}

NB_MODULE(_quantlop, module_py)
{
    module_py.doc() = "Quantlop C++ core bindings";

    class_<PauliWord>(module_py, "_PauliWord")
        .def(init<Complex, String>(), arg("coeff"), arg("string"))
        .def("_num_qubits", &PauliWord::num_qubits)
        .def("_get_coeff", &PauliWord::coeff)
        .def("_get_string", &PauliWord::string);

    class_<Hamiltonian>(module_py, "_Hamiltonian")
        .def(init<std::vector<PauliWord>>(), arg("pwords"))
        .def("_num_qubits", &Hamiltonian::num_qubits)
        .def("_num_terms", &Hamiltonian::num_terms)
        .def("_get_pwords", &Hamiltonian::terms);

    module_py.def(
        "_evolve_higham",
        &evolve_py<evolve_higham>,
        arg("ham"),
        arg("psi"),
        arg("theta"),
        arg("rtol"),
        arg("num_threads"));

    module_py.def(
        "_evolve_krylov",
        &evolve_py<evolve_krylov>,
        arg("ham"),
        arg("psi"),
        arg("theta"),
        arg("rtol"),
        arg("num_threads"));
}
