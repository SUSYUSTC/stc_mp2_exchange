# Quadrature grids for the Laplace representation of the energy denominator,
#
#     1 / Delta = sum_k weight_k exp(-beta_k Delta),   Delta in [Delta_min, Delta_max]
#
# Two choices. Gauss-Legendre in a substituted variable is the naive one: it knows
# only Delta_min, so it cannot adapt to how wide the denominator range actually is.
# Minimax fits 1/x on the true range and is several orders of magnitude better at
# the same number of points.
#
# numpy only, so this stays importable without dragging in torch/numba.
import os
import re

import numpy as np

this_dir = os.path.dirname(os.path.abspath(__file__))
minimax_table_file = os.path.join(this_dir, 'data', 'init_para.txt')

# Scale of the Gauss-Legendre substitution t = exp(-a Delta_min beta). A fudge
# factor standing in for the range information that scheme does not have.
a = 0.5

_minimax_table = {}


def get_gauss_legendre_quadrature(Delta_min, M, a=a):
    nodes, weights = np.polynomial.legendre.leggauss(M)
    t = 0.5 * (nodes + 1)
    w = 0.5 * weights
    return [(-np.log(x) / (a * Delta_min), y / (a * Delta_min * x)) for x, y in zip(t, w)]


def read_minimax_table(path=minimax_table_file):
    # Best exponential-sum approximations of 1/x on [1, R], keyed by (M, R):
    #   1/x = sum_k omega_k exp(-alpha_k x)
    # Hackbusch's tables as shipped with laplace-minimax; see data/README.md.
    if _minimax_table:
        return _minimax_table
    key = None
    values = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        head = re.match(r'1_xk(\d+)_(\S+)$', line)
        if head:
            if key is not None:
                _minimax_table[key] = np.array(values)
            token = head.group(2)
            # Two encodings live in the file: 'dEn' -> d*10^n, 'dddd' -> d + ddd/1000.
            if token[1:2] == 'E':
                R = float(token[0]) * 10.0 ** int(token[2:])
            else:
                R = int(token[0]) + int(token[1:4]) / 1000.0
            key = (int(head.group(1)), R)
            values = []
        elif not line.startswith('$'):
            values.append(float(line.split()[0]))
    if key is not None:
        _minimax_table[key] = np.array(values)
    return _minimax_table


def minimax_error(x, omega, alpha):
    # Absolute error. This is the norm to minimize: the MP2 error is
    # sum_ijab N_ijab (1/D - sum_k w_k exp(-b_k D)) with every N positive, so it is
    # the absolute error weighted by numerators. Minimizing the relative error
    # 1 - x*sum instead gives a ~10x smaller relative error and no gain in energy.
    return 1.0 / x - (np.exp(-np.outer(x, alpha)) * omega).sum(1)


def minimax_error_derivative(x, omega, alpha):
    return -1.0 / x**2 + (np.exp(-np.outer(x, alpha)) * omega * alpha).sum(1)


def find_minimax_extrema(omega, alpha, R, nscan=40000):
    x = np.logspace(0, np.log10(R), nscan)
    derivative = minimax_error_derivative(x, omega, alpha)
    extrema = [1.0]
    for i in np.nonzero(np.sign(derivative[:-1]) != np.sign(derivative[1:]))[0]:
        lo, hi = x[i], x[i + 1]
        for _ in range(90):
            mid = np.sqrt(lo * hi)
            if np.sign(minimax_error_derivative(np.array([lo]), omega, alpha)[0]) == \
               np.sign(minimax_error_derivative(np.array([mid]), omega, alpha)[0]):
                lo = mid
            else:
                hi = mid
        extrema.append(np.sqrt(lo * hi))
    return np.array(extrema + [R])


def remez_minimax(omega, alpha, R, niter=60, tol=1e-15, equioscillation_tol=1e-7):
    # Newton on the equioscillation conditions err(x_i) = (-1)^i eps at the 2M+1
    # extrema, alternating with a refresh of the extrema themselves.
    M = len(omega)
    for iteration in range(niter):
        extrema = find_minimax_extrema(omega, alpha, R)
        if len(extrema) != 2 * M + 1:
            raise RuntimeError(f'minimax Remez at R={R:.6g}, M={M}: found {len(extrema)} '
                               f'extrema, expected {2 * M + 1}')
        sign = (-1.0) ** np.arange(2 * M + 1)
        if sign[0] * minimax_error(extrema, omega, alpha)[0] < 0:
            sign = -sign
        eps = np.mean(np.abs(minimax_error(extrema, omega, alpha)))
        for _ in range(80):
            E = np.exp(-np.outer(extrema, alpha))
            residual = minimax_error(extrema, omega, alpha) - sign * eps
            jacobian = np.empty((2 * M + 1, 2 * M + 1))
            jacobian[:, :M] = -E
            jacobian[:, M:2 * M] = extrema[:, None] * E * omega
            jacobian[:, 2 * M] = -sign
            step = np.linalg.solve(jacobian, -residual)
            damping = 1.0
            while damping > 1e-6:
                omega_new = omega + damping * step[:M]
                alpha_new = alpha + damping * step[M:2 * M]
                if np.all(alpha_new > 0) and np.all(omega_new > 0):
                    break
                damping *= 0.5
            else:
                raise RuntimeError(f'minimax Remez at R={R:.6g}, M={M}: Newton step left '
                                   f'the positive orthant')
            omega, alpha = omega_new, alpha_new
            eps = eps + damping * step[2 * M]
            if np.max(np.abs(step[:2 * M])) < tol * max(1.0, np.max(np.abs(alpha))):
                break
        achieved = np.max(np.abs(minimax_error(np.logspace(0, np.log10(R), 200000), omega, alpha)))
        if abs(achieved - abs(eps)) / achieved < equioscillation_tol:
            return omega, alpha, achieved
    raise RuntimeError(f'minimax Remez at R={R:.6g}, M={M}: no equioscillation after '
                       f'{niter} iterations (float64 runs out around M=10-12; the '
                       f'reference library uses double-double there)')


def get_minimax_quadrature(Delta_min, Delta_max, M, nstep=16):
    # The tabulated grids are exact minimax solutions only at discrete R, so the
    # table entry is a starting guess: Remez then refines it to the true R. Skipping
    # that costs a factor 7-14 in energy error even for a 1.24x overshoot. Refining
    # in one jump would push the outermost oscillations off the end of the shortened
    # interval and lose extrema, so walk R down in geometric steps.
    #
    # Returns the grid and the achieved absolute error of the 1/x fit, which bounds
    # the energy bias once weighted by the (positive) numerators.
    table = read_minimax_table()
    R = Delta_max / Delta_min
    available = sorted(r for M_tab, r in table if M_tab == M and r >= R)
    if not available:
        possible = sorted(M_tab for M_tab, r in table if r >= R)
        raise RuntimeError(f'no tabulated minimax grid for M={M}, R={R:.4g}; '
                           f'available M for this R: {possible[0]}..{possible[-1]}')
    parameters = table[(M, available[0])]
    omega, alpha = parameters[:M].copy(), parameters[M:].copy()
    for R_step in np.geomspace(available[0], R, nstep)[1:]:
        omega, alpha, error = remez_minimax(omega, alpha, R_step)
    # Tabulated on [1, R]; the whole problem scales by 1/Delta_min.
    beta_weight = [(al / Delta_min, om / Delta_min) for om, al in zip(omega, alpha)]
    return beta_weight, error
