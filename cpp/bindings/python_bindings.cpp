#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <quantlop/evolution.hpp>
#include <quantlop/hamiltonian.hpp>
#include <quantlop/pauliword.hpp>
#include <quantlop/types.hpp>

using ComplexArray = nanobind::ndarray<const Complex, nanobind::ndim<1>, nanobind::c_contig, nanobind::device::cpu>;
using NumpyComplexArray = nanobind::ndarray<nanobind::numpy, Complex, nanobind::ndim<1>, nanobind::c_contig>;

static NumpyComplexArray evolve_py(const Hamiltonian &ham, ComplexArray psi, Complex theta, int num_threads)
{
    const Size dim = psi.shape(0);
    Complex *out_ptr = evolve(ham, psi.data(), theta, num_threads);
    nanobind::capsule owner(out_ptr, [](void *p) noexcept { delete[] static_cast<Complex *>(p); });
    return NumpyComplexArray(out_ptr, {dim}, owner);
}

NB_MODULE(_quantlop, module_py)
{
    module_py.doc() = "Quantlop C++ core bindings";

    nanobind::class_<PauliWord>(module_py, "_PauliWord")
        .def(nanobind::init<Complex, String>(), nanobind::arg("coeff"), nanobind::arg("string"))
        .def("_num_qubits", &PauliWord::num_qubits)
        .def("_get_coeff", &PauliWord::coeff)
        .def("_get_string", &PauliWord::string);

    nanobind::class_<Hamiltonian>(module_py, "_Hamiltonian")
        .def(nanobind::init<std::vector<PauliWord>>(), nanobind::arg("pwords"))
        .def("_num_qubits", &Hamiltonian::num_qubits)
        .def("_num_terms", &Hamiltonian::num_terms)
        .def("_get_pwords", &Hamiltonian::terms);

    module_py.def(
        "_evolve",
        &evolve_py,
        nanobind::arg("ham"),
        nanobind::arg("psi"),
        nanobind::arg("theta"),
        nanobind::arg("num_threads"));
}
