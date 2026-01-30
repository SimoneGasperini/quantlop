from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply, funm_multiply_krylov


def evolve(ham, psi, coeff=1, method="higham"):
    linop = -1j * coeff * ham
    if method == "higham":
        return expm_multiply(linop, psi, traceA=0)
    elif method == "krylov":
        return funm_multiply_krylov(expm, linop, psi)
    else:
        raise NotImplementedError
