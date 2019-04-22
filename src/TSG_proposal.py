import matplotlib.pyplot as plt
import numpy as np
import math
import h5py
import torch


def f_Y(y_init, a, b, x, timesteps):
    y_prev = y_init
    y_x = []

    for i, t, x_k in enumerate(zip(timesteps, x)):
        if i == 0:
            t_prev = t
            continue
        dt = t - t_prev
        y_x.append(y_prev + dY_dx(a, b, x_k) * dt)
        y_prev = y_x[-1]
        t_prev = t

    return np.array(y_x)
    # return y_prev + dY_dx(a, b, x) * dt


# def f_Y_torch(a, b, x):
#     return f_Y(a, b, x)


def dY_dx(a, b, x):  # where x = cons(x_0)
    # return a*x**3 + b * x**2
    return a * math.cos(x) ** 3 + b * math.sin(x) ** 2
    # return -a * torch.sin(2*x) - b * torch.sin(x)


def f_X(t):
    return math.cos(t)


def f_X_inv(x):
    return torch.acos(x)


def get_func_timeseries(f_X, f_Y, a, b, diapasone=(0, math.pi * 100, math.pi / 30)):
    """Function generates timeserise, used for testing"""
    T = [t for t in np.arange(diapasone[0], diapasone[1], diapasone[2])]
    x_t = np.array([f_X(t) for t in T])
    y_x = f_Y(y_init=1, a=a, b=b, x=x_t, diapasone=diapasone)
    y_x = np.array([f_Y(a, b, x, dY_dx) for x in x_t])

    plt.plot(T, y_x)
    plt.plot(T, x_t)
    return x_t, y_x
