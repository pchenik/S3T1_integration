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

    x = abs((math.log(abs((s1 - s0) / (s1 - s2)))) / math.log(L))
    #x = -(math.log(abs((s2 - s1) / (s1 - s0)))) / math.log(L)
    # if x is None or abs(x - 4) > 2:
    #     print(s0, ' ', s1, ' ', s2, ' ', L)
    return x
    #not sure about floor

    raise NotImplementedError


def quad(func, x0, x1, xs, **kwargs):
    """
    func: function to integrate
    x0, x1: interval to integrate on
    xs: nodes
    **kwargs passed to moments()
    """

    CONST = 10 ** 6
    s = len(xs)

    X = np.array([[xs[j] ** i for j in range(s)] for i in range(s)])
    #print(kwargs)
    u = np.array(moments(s - 1, x0, x1, a=kwargs.setdefault('a', 0), b=kwargs.setdefault('b', 0),
                         alpha=kwargs.setdefault('alpha', 0), beta=kwargs.setdefault('beta', 0)))
    #print(u)
    # if np.linalg.det(X) == 0:
    #     print(X)

    # X = X * CONST
    # u = u * CONST
    A = np.linalg.solve(X, u)
    #A, rnorm = nnls(X, u)
    #print(A)

    ans = 0
    for i in range(0, s):
        ans += A[i] * func(xs[i])

    return ans

    raise NotImplementedError


def newton_cotes(func, x0, x1, n, **kwargs):
    u = moments(n, x0, x1, a=kwargs.setdefault('a', 0), b=kwargs.setdefault('b', 0),
                alpha=kwargs.setdefault('alpha', 0), beta=kwargs.setdefault('beta', 0))
    d = np.array([-n * u[i] / u[0] for i in range(1, n + 1)])
    U = [[0 for j in range(n)] for i in range(n)]
    for i in range(1, n):
        for j in range(i):
            U[i][j] = n * u[i - j] / u[0]

    for i in range(n):
        U[i][i] = i + 1

    U = np.array(U)

    a = np.linalg.solve(U, d)

    a = np.insert(a, 0, 1)

    xs = np.roots(a)
    Cn = u[0] / n
    ans = 0
    for i in range(n):
        ans += func(xs[i])

    return Cn * ans


def quad_s(f, x, n):
    h = (x[1] - x[0]) / n
    xs = np.linspace(x[0], x[1], n+1)
    xs_m = xs[:-1] + h/2
    fs = f(xs)
    fs_m = f(xs_m)
    return (np.sum(fs[:-1]) + 4*np.sum(fs_m) + np.sum(fs[1:])) * h /6

def quad_gauss(func, x0, x1, n, **kwargs):
    """
    func: function to integrate
    x0, x1: interval to integrate on
    n: number of nodes
    """

    CONST = 10 ** 6

    u = moments(2 * n - 1, x0, x1, a=kwargs.setdefault('a', 0), b=kwargs.setdefault('b', 0),
    alpha=kwargs.setdefault('alpha', 0), beta=kwargs.setdefault('beta', 0))


    U = np.array([[u[j + i] for j in range(n)] for i in range(n)])

    b = np.array([-u[n + i] for i in range(n)])
    # U = U * CONST
    # b = b * CONST

    if np.linalg.det(U) == 0:
        # print(np.linalg.det(U))
        # print(b)
        # print(u)
        # print(x0, ' ', x1, ' ', n, ' ', kwargs)
        # print(U)
        # print('done\n')
        #return quad_gauss(func, x0, x1, n - 1, **kwargs)
        return newton_cotes(func, x0, x1, n ** 3, **kwargs)
        #return quad_s(func, [x0, x1], n ** 5)
        #return quad(func, x0, x1, [x0 + i * ((x1 - x0) / (n ** 2 + 1)) for i in range(n ** 2 + 2)], **kwargs)
        # return (x1 - x0) * func((x0 + x1) / 2)


    a = np.linalg.solve(U, b)
    #a, rnorm = nnls(U, b)


    a = np.append(a, [1])
    coeff = a[::-1]

    xs = np.roots(coeff)

    #print(xs)

    xs = np.sort(xs)
    for x in xs:
     if np.iscomplex(x):
       print("KEK")

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
    eps = 1e-9
    x = x0

    while x + eps < x1:
        ans += newton_cotes(func, x, x + h, n_nodes, **kwargs)
        # ans += quad(func, x, x + h, [x + i * (h / (n_nodes + 1)) for i in range(n_nodes + 2)], **kwargs)
        #ans += quad_gauss(func, x, x + h, n_nodes, **kwargs)
        x += h

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
    ans = 0.5

    while (eps > tol):
        h1, h2, h3 = h, h / L, h / L ** 2

        s0 = composite_quad(func, x0, x1, round((x1 - x0) / h1), n_nodes, alpha=0, beta=0)
        s1 = composite_quad(func, x0, x1, round((x1 - x0) / h2), n_nodes, alpha=0, beta=0)
        s2 = composite_quad(func, x0, x1, round((x1 - x0) / h3), n_nodes, alpha=0, beta=0)
        m = aitken(s0, s1, s2, L)
        eps = max(runge(s0, s1, m, L))
        print(eps)

        h = h1 * (tol / abs(s0)) ** (1 / m)
        h *= 0.95

    return s1, eps

    raise NotImplementedError


# x0, x1 = 1, 3
# alpha = 0.14
# beta = 0.88
# p = Monome(1)
# xs = np.linspace(x0, x1, 6)[1:-1]
# print(xs)
# res = quad(p, x0, x1, xs, a=x0, alpha=alpha)
# print(res)

#print(composite_quad(lambda x: x * x, 0, 1, 1000, 3, alpha=0, beta=0))
#print(integrate(lambda x: math.sin(x), 0, math.pi, 0.0000000001))

# x0, x1 = 0, 1
# L = 2
# n_intervals = [L ** q for q in range(0, 8)]
# p = Monome(5)
# Y = [composite_quad(p, x0, x1, n_intervals=n, n_nodes=5) for n in n_intervals]
# print(Y)
# #exact = sp_quad(lambda x: p(x) / (x)**0.83, x0, x1)[0]
# #accuracy = get_log_error(Y, exact * np.ones_like(Y))
# accuracy = get_log_error(Y, p[x0, x1] * np.ones_like(Y))
# print(p[x0, x1] * np.ones_like(Y))
# print(p[x0, x1])
# print(accuracy)



# a, b, alpha, beta, f = params(6)
# x0, x1 = a, b
# # a, b = -10, 10
# print(a)
# print(b)
# print(alpha)
# print(beta)
# exact = sp_quad(lambda x: f(x) / (x-a)**alpha / (b-x)**beta, x0, x1)[0]
# print(exact)
# # plot weights
# xs = np.linspace(x0, x1, 101)[1:-1]
# print(xs)
# ys = 1 / ((xs - a) ** alpha * (b - xs) ** beta)
# print(ys)
#
# L = 2
# n_intervals = [L ** q for q in range(2, 10)]
# n_nodes = 3
# Y = [composite_quad(f, x0, x1, n_intervals=n, n_nodes=n_nodes,
#                     a=a, b=b, alpha=alpha, beta=beta) for n in n_intervals]
# accuracy = get_log_error(Y, exact * np.ones_like(Y))
# x = np.log10(n_intervals)
# aitken_degree = aitken(*Y[5:8], L)




# x0, x1 = 0, 1
# max_degree = 7
# p = Monome(5)
# y0 = p[x0, x1]
# max_node_count = range(1, max_degree+1)
# Y = [quad(p, x0, x1, np.linspace(x0, x1, node_count)) for node_count in max_node_count]
# accuracy = get_log_error(Y, y0 * np.ones_like(Y))

#print(moments(3, 1.3, 2.2, 1.3, 2.2, beta=0.833))