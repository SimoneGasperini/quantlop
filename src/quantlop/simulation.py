from functools import cache
import numpy as np


def evolve(ham, psi, coeff=1.0):
    expm = -1j * coeff * ham
    return _expm_multiply(expm, psi)


def _expm_multiply(A, b):
    # Algorithm (3.2) in https://doi.org/10.1137/100788860 with balance=False, n0=1, t=1, trace(A)=0
    tol = 2**-53
    m_star, s = _fragment_3_1(A)
    f = b
    for _ in range(s):
        c1 = np.linalg.norm(b, ord=np.inf)
        for j in range(1, m_star + 1):
            b = A._matvec(b) / float(s * j)
            c2 = np.linalg.norm(b, ord=np.inf)
            f = f + b
            if c1 + c2 <= tol * np.linalg.norm(f, ord=np.inf):
                break
            c1 = c2
        b = f
    return f


def _fragment_3_1(A, p_max=8, m_max=55):
    m_star = None
    s = None
    A_one_norm = _one_norm(A)
    # Condition (3.13) in https://doi.org/10.1137/100788860 with l=1, n0=1
    if A_one_norm <= 2 * _theta[m_max] / m_max * p_max * (p_max + 3):
        for m in _theta:
            s_m = np.ceil(A_one_norm / _theta[m])
            if m_star is None or m * s_m < m_star * s:
                m_star = m
                s = s_m
    else:
        # Equation (3.11) in https://doi.org/10.1137/100788860
        for p in range(2, p_max + 1):
            alpha_p = max(compute_d(A, p), compute_d(A, p + 1))
            for m in range(p * (p - 1) - 1, m_max + 1):
                s_m = np.ceil(alpha_p / _theta[m])
                if m_star is None or m * s_m < m_star * s:
                    m_star = m
                    s = s_m
        s = max(s, 1)
    return int(m_star), int(s)


def _one_norm(A):
    n_cols = A.shape[1]
    e = np.zeros(n_cols)
    max_col_sum = 0
    for j in range(n_cols):
        e[j] = 1
        col_sum = np.sum(np.abs(A.matvec(e)))
        if col_sum > max_col_sum:
            max_col_sum = col_sum
        e[j] = 0
    return max_col_sum


@cache
def compute_d(A, p):
    return _one_norm(A**p) ** (1 / p)


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
