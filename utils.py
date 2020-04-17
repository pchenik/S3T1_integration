import numpy as np


def call_counter(func):
    def helper(*args, **kwargs):
        helper.calls += 1
        return func(*args, **kwargs)
    helper.calls = 0
    return helper


def get_log_error(x, y, axis=1):
    x = np.array(x)
    y = np.array(y)
    if len(x.shape) == 1:
        #temp = -np.log10(np.abs(x - y))
        return -np.log10(np.abs(x - y))
    else:
        return -np.log10(np.linalg.norm(x - y, axis=axis))
