import matplotlib.pyplot as plt
import numpy as np
import math 
import h5py
import torch


def f_X(t):
    return math.sin(t)


def f_X_torch(t):
    return torch.sin(t)


def f_Y(a, b, x):
    return a * x**2 + b * x


def dY_dx(a, b, x):
    return a * 2 * x + b



def f_X_inv(x):
    return torch.asin(x)


def get_func_timeseries(f_X, f_Y, a, b, diapasone = (0, math.pi * 100, math.pi / 30)):
    """Function generates timeserise, used for testing"""
    T = [t for t in np.arange(diapasone[0], diapasone[1], diapasone[2])]
    x_t = np.array([f_X(t) for t in T])

    y_x = np.array([f_Y(a, b, x) for x in x_t])

    plt.plot(T, y_x)
    plt.plot(T, x_t)
    return x_t, y_x
