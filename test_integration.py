import pytest
import numpy as np
from scipy.integrate import quad as sp_quad
import matplotlib.pyplot as plt

from integrate_collection import Monome, Harmonic

from utils import get_log_error
from integration import (quad,
                            quad_gauss,
                            composite_quad,
                            integrate,
                            aitken,
                            moments)


def test_quad_degree():
    """
    check quadrature degree
    Q: why in some cases x^n integrated perfectly with only n nodes?
    """
    x0, x1 = 0, 1

    max_degree = 7

    for deg in range(1, max_degree):
        p = Monome(deg)
        y0 = p[x0, x1]

        max_node_count = range(1, max_degree+1)

        Y = [quad(p, x0, x1, np.linspace(x0, x1, node_count)) for node_count in max_node_count]
        # Y = [quad(p, x0, x1, x0 + (x1-x0) * np.random.random(node_count)) for node_count in max_node_count]
        accuracy = get_log_error(Y, y0 * np.ones_like(Y))
        accuracy[np.isinf(accuracy)] = 17

        # check accuracy is good enough
        for node_count, acc in zip(max_node_count, accuracy):
            if node_count >= deg + 1:
                assert acc > 6

        plt.plot(max_node_count, accuracy, '.:', label=f'x^{deg}')

    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('node_count')
    plt.suptitle(f'test quad')
    plt.show()


def test_weighted_quad_degree():
    """
    check weighted quadrature degree
    we compare n-th moment of weight function calculated in two ways:
        - by moments()
        - numerically by quad()
    """
    x0, x1 = 1, 3
    alpha = 0.14
    beta = 0.88

    max_degree = 7
    for deg in range(1, max_degree):
        p = Monome(deg)
        xs = np.linspace(x0, x1, 6)[1:-1]  # 4 points => accuracy degree is 3

        res = quad(p, x0, x1, xs, a=x0, alpha=alpha)
        ans = moments(deg, x0, x1, a=x0, alpha=alpha)[-1]
        d = abs(res - ans)
        print(f'{deg:2}-a: {res:8.3f} vs {ans:8.3f}, delta = {d:e}')
        if deg < len(xs):
            assert d < 1e-6

        res = quad(p, x0, x1, xs, b=x1, beta=beta)
        ans = moments(deg, x0, x1, b=x1, beta=beta)[-1]
        d = abs(res - ans)
        print(f'{deg:2}-b: {res:8.3f} vs {ans:8.3f}, delta = {d:e}')
        if deg < len(xs):
            assert d < 1e-6


def test_quad_gauss_degree():
    """
    check gaussian quadrature degree
    """
    x0, x1 = 0, 1

    max_degree = 8

    for deg in range(2, max_degree):
        p = Monome(deg)
        y0 = p[x0, x1]

        max_node_count = range(2, 6)
        Y = [quad_gauss(p, x0, x1, node_count) for node_count in max_node_count]
        accuracy = get_log_error(Y, y0 * np.ones_like(Y))
        accuracy[np.isinf(accuracy)] = 17

        # check accuracy is good enough
        for node_count, acc in zip(max_node_count, accuracy):
            if 2 * node_count >= deg + 1:
                assert acc > 6

        plt.plot(max_node_count, accuracy, '.:', label=f'x^{deg}')

    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('node count')
    plt.suptitle(f'test quad gauss')
    plt.show()


@pytest.mark.parametrize('n_nodes', [2, 3])
def test_composite_quad(n_nodes):
    """
    test composite 2-, 3-, 5-node quads
    Q: explain converge speed for each case
    """
    plt.figure()

    x0, x1 = 0, 1
    L = 2
    n_intervals = [L ** q for q in range(0, 8)]

    for i, degree in enumerate((5, 6)):
        p = Monome(degree)
        Y = [composite_quad(p, x0, x1, n_intervals=n, n_nodes=n_nodes) for n in n_intervals]
        accuracy = get_log_error(Y, p[x0, x1] * np.ones_like(Y))
        x = np.log10(n_intervals)

        # check convergence
        ind = np.isfinite(x) & np.isfinite(accuracy)
        k, b = np.polyfit(x[ind], accuracy[ind], 1)
        aitken_degree = aitken(*Y[0:6:2], L ** 2)

        plt.subplot(1, 2, i+1)
        plt.title(f'{n_nodes}-node CQ for x^{degree}')
        plt.plot(x, k*x+b, 'b:', label=f'{k:.2f}*x+{b:.2f}')
        plt.plot(x, aitken_degree*x+b, 'm:', label=f'aitken ({aitken_degree:.2f})')
        plt.plot(x, accuracy, 'kh', label=f'accuracy for x^{degree}')
        plt.xlabel('log10(node count)')
        plt.ylabel('accuracy')
        plt.legend()

        if n_nodes < degree:
            assert np.abs(aitken_degree - k) < 0.5, \
                f'Aitken estimation {aitken_degree:.2f} is too far from actual {k:.2f}'

    plt.show()


@pytest.mark.parametrize('v', [2, 3, 5, 6])
def test_composite_quad_degree(v):
    """
    Q: convergence maybe somewhat between 3 and 4, why?
    """
    from variants import params

    plt.figure()
    a, b, alpha, beta, f = params(v)
    x0, x1 = a, b
    # a, b = -10, 10
    exact = sp_quad(lambda x: f(x) / (x-a)**alpha / (b-x)**beta, x0, x1)[0]

    # plot weights
    xs = np.linspace(x0, x1, 101)[1:-1]
    ys = 1 / ((xs - a) ** alpha * (b - xs) ** beta)

    #my addition
    # for x in xs:
    #     if (x-a)**alpha * (b-x)**beta:
    #         print("HERE")
    # print(ys)
    #

    plt.subplot(1, 2, 1)
    plt.plot(xs, ys, label='weights')
    ax = list(plt.axis())
    ax[2] = 0.
    plt.axis(ax)
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.legend()

    L = 2
    n_intervals = [L ** q for q in range(2, 10)]
    n_nodes = 3
    Y = [composite_quad(f, x0, x1, n_intervals=n, n_nodes=n_nodes,
                        a=a, b=b, alpha=alpha, beta=beta) for n in n_intervals]
    accuracy = get_log_error(Y, exact * np.ones_like(Y))
    x = np.log10(n_intervals)
    aitken_degree = aitken(*Y[5:8], L)

    # plot acc
    plt.subplot(1, 2, 2)
    plt.plot(x, accuracy, 'kh')
    plt.xlabel('log10(node count)')
    plt.ylabel('accuracy')
    plt.suptitle(f'variant #{v} (alpha={alpha:4.2f}, beta={beta:4.2f})\n'
                 f'aitken estimation: {aitken_degree:.2f}')
    plt.show()


def test_integrate():
    """
    integrate with a given tolerance
    """
    p = Harmonic(1, 0)
    x0, x1 = 0, np.pi

    for tol in 10. ** np.arange(-9, -2):
        s, err = integrate(p, x0, x1, tol=tol)

        print(f'Check for tol {tol:.2e}: res = {s-err:.6f} .. {s:.6f} .. {s+err:.6f}')

        assert err >= 0,                            'estimated error should be >= 0'
        assert np.abs(p[x0, x1] - s) <= 1.1*err,    'actual error should be <= estimated error + 10%'
        assert np.abs(p[x0, x1] - s) <= tol,        'actual error should be <= tolerance'
