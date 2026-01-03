import numpy as np


def evolve(ham, psi, coeff=1):
    expm = -1j * coeff * ham
    return _expm_multiply(expm, psi)


def _expm_multiply(A, b):
    # Algorithm (3.2) in https://doi.org/10.1137/100788860 with balance=False, n0=1, t=1, trace(A)=0
    tol = 2**-53
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


_theta = {
    # Table (A.1) in https://doi.org/10.1017/S0962492910000036
    1: 2.29e-16,
    2: 2.58e-8,
    3: 1.39e-5,
    4: 3.40e-4,
    5: 2.40e-3,
    6: 9.07e-3,
    7: 2.38e-2,
    8: 5.00e-2,
    9: 8.96e-2,
    10: 1.44e-1,
    11: 2.14e-1,
    12: 3.00e-1,
    13: 4.00e-1,
    14: 5.14e-1,
    15: 6.41e-1,
    16: 7.81e-1,
    17: 9.31e-1,
    18: 1.09,
    19: 1.26,
    20: 1.44,
    21: 1.62,
    22: 1.82,
    23: 2.01,
    24: 2.22,
    25: 2.43,
    26: 2.64,
    27: 2.86,
    28: 3.08,
    29: 3.31,
    30: 3.54,
    # Table (3.1) in https://doi.org/10.1137/100788860 (double precision)
    35: 4.7,
    40: 6.0,
    45: 7.2,
    50: 8.5,
    55: 9.9,
}
