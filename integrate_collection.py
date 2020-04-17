import numpy as np


class FunctionToIntegrate():
    """
    base class
    self(x) for evaluation
    self[x0, x1] for getting integral
    """
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def indefinite(self, x):
        raise NotImplementedError

    def __getitem__(self, item):
        assert '__len__' in dir(item), \
            f'{self.__class__}: index should have 2 items'
        assert len(item) == 2, \
            f'{self.__class__}: index should have 2 items'
        x0, x1 = item
        return self.indefinite(x1) - self.indefinite(x0)


class Monome(FunctionToIntegrate):
    """
    x ** degree
    """
    def __init__(self, degree):
        self.degree = degree
        super().__init__()

    def __call__(self, x):
        return x ** self.degree

    def indefinite(self, x):
        k = self.degree + 1
        return x ** k / k


class Harmonic(FunctionToIntegrate):
    """
    a * sin(x) + b * cos(x)
    """
    def __init__(self, a, b):
        self.a = a
        self.b = b
        super().__init__()

    def __call__(self, x):
        return self.a * np.sin(x) + self.b * np.cos(x)

    def indefinite(self, x):
        return -self.a * np.cos(x) + self.b * np.sin(x)
