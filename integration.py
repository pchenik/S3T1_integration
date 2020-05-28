import numpy as np
import math
from variants import params
from scipy.optimize import nnls
from scipy.integrate import quad as sp_quad
from integrate_collection import Monome, Harmonic
from utils import get_log_error

def moments(max_s, xl, xr, a=None, b=None, alpha=0.0, beta=0.0): #Test OK
    """
    compute 0..max_s moments of the weight p(x) = 1 / (x-a)^alpha / (b-x)^beta over [xl, xr]
    """

    def moment_a(a, s, alpha):
        series = 0
        multiplier = 1

        for i in range(0, s + 1):
            multiplier *= 1 / (-alpha + i + 1)
            F = lambda x: (-1) ** i * (x + a) ** (s - i) * multiplier * x ** (-alpha + i + 1)
            series += F(xr - a) - F(xl - a)
            multiplier *= (s - i)

        return series


    def moment_b(b, s, beta):
        series = 0
        multiplier = 1

        for i in range(0, s + 1):
            multiplier /= (-beta + i + 1)
            F = lambda x: (b - x) ** (s - i) * multiplier * x ** (-beta + i + 1)
            series += F(b - xr) - F(b - xl)
            multiplier *= (s - i)

        return series

    assert alpha * beta == 0, f'alpha ({alpha}) and/or beta ({beta}) must be 0'
    if alpha != 0.0:
        assert a is not None, f'"a" not specified while alpha != 0'
        return [(moment_a(a, s, alpha)) for s in range(0, max_s + 1)]

    if beta != 0.0:
        assert b is not None, f'"b" not specified while beta != 0'
        return [-(moment_b(b, s, beta)) for s in range(0, max_s + 1)]

    if alpha == 0 and beta == 0:
        return [(xr ** s - xl ** s) / s for s in range(1, max_s + 2)]

    raise NotImplementedError


def runge(s0, s1, m, L):
    """
    estimate m-degree errors for s0 and s1
    """
    d0 = np.abs(s1 - s0) / (1 - L ** -m)
    d1 = np.abs(s1 - s0) / (L ** m - 1)
    return d0, d1

def aitken(s0, s1, s2, L):
    """
    estimate accuracy degree
    s0, s1, s2: consecutive composite quads
    return: accuracy degree estimation
    """

    x = abs((math.log(abs((s1 - s0) / (s1 - s2)))) / np.log(L))
    return x

    raise NotImplementedError


def quad(func, x0, x1, xs, **kwargs):
    """
    func: function to integrate
    x0, x1: interval to integrate on
    xs: nodes
    **kwargs passed to moments()
    """

    s = len(xs)

    X = np.vander(xs, increasing=True).T
    u = np.array(moments(s - 1, x0, x1, **kwargs))

    A = np.linalg.solve(X, u)

    ans = 0
    for i in range(0, s):
        ans += A[i] * func(xs[i])

    return ans

    raise NotImplementedError

def quad_gauss(func, x0, x1, n, **kwargs):
    """
    func: function to integrate
    x0, x1: interval to integrate on
    n: number of nodes
    """

    u = moments(2 * n - 1, x0, x1, **kwargs)
    U = np.array([[u[j + i] for j in range(n)] for i in range(n)])
    b = np.array([-u[n + i] for i in range(n)])

    a = np.linalg.solve(U, b)
    a = np.append(a, [1])
    coeff = a[::-1]
    xs = np.roots(coeff)
    xs = np.sort(xs)
    return quad(func, x0, x1, xs, **kwargs)

    raise NotImplementedError

def composite_quad(func, x0, x1, n_intervals, n_nodes, **kwargs):
    """
    func: function to integrate
    x0, x1: interval to integrate on
    n_intervals: number of intervals
    n_nodes: number of nodes on each interval
    """

    h = (x1 - x0) / n_intervals
    ans = 0
    xs = np.linspace(x0, x1, n_intervals + 1)

    for i, x in enumerate(xs[:-1]):
        ans += quad(func, x, x + h, np.linspace(x, x + h, n_nodes), **kwargs)

    return ans

    raise NotImplementedError


def integrate(func, x0, x1, tol):
    """
    integrate with error <= tol
    return: result, error estimation
    """

    h = x1 - x0
    L = 2
    s0 = s1 = s2 = 0
    eps = 1
    n_nodes = 3

    while eps > tol:
        h1, h2, h3 = h, h / L, h / L ** 2

        s0 = composite_quad(func, x0, x1, round((x1 - x0) / h1), n_nodes, alpha=0, beta=0)
        s1 = composite_quad(func, x0, x1, round((x1 - x0) / h2), n_nodes, alpha=0, beta=0)
        s2 = composite_quad(func, x0, x1, round((x1 - x0) / h3), n_nodes, alpha=0, beta=0)
        m = aitken(s0, s1, s2, L)
        eps = max(runge(s0, s1, m, L))

        h = h1 * (tol / abs(s0)) ** (1 / m)
        h *= 0.95

    return s0, eps

    raise NotImplementedError
