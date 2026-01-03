import numpy as np


def evolve(ham, psi, coeff=1):
    expm = -1j * coeff * ham
    return _expm_multiply(expm, psi)


def _expm_multiply(A, b):
    # Algorithm (3.2) in https://doi.org/10.1137/100788860 with balance=False, n0=1, t=1, trace(A)=0
    tol = 2**-24
    m_star, s = _fragment_3_1(A)
    f = b
    for _ in range(s):
        c1 = np.max(np.abs(b))
        for j in range(1, m_star + 1):
            b = A._matvec(b) / (s * j)
            c2 = np.max(np.abs(b))
            f = f + b
            if c1 + c2 <= tol * np.max(np.abs(f)):
                break
            c1 = c2
        b = f
    return f


def _fragment_3_1(A):
    m_star = None
    s = None
    A_one_norm = A.lcu_norm()
    for m, theta in _theta.items():
        s_m = np.ceil(A_one_norm / theta)
        if m_star is None or m * s_m < m_star * s:
            m_star = m
            s = s_m
    return int(m_star), int(s)


# Table (3.1) in https://eprints.maths.manchester.ac.uk/1591/1/alhi11.pdf
_theta = {
    5: 1.3e-1,
    10: 1.0e0,
    15: 2.2e0,
    20: 3.6e0,
    25: 4.9e0,
    30: 6.3e0,
    35: 7.7e0,
    40: 9.1e0,
    45: 1.1e1,
    50: 1.2e1,
    55: 1.3e1,
}
