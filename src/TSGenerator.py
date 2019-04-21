import matplotlib.pyplot as plt
import numpy as np
import math 
import h5py
import torch

def f_Y(a, t):
    return math.cos(a*t)
def dY_dt(a, t):
    return -a * torch.sin(a*t)
def f_X(t):
    return math.sin(t)
def f_X_inv(x):
    return torch.asin(x)

def get_func_timeseries(f_X, f_Y, a = -1, diapasone = (0, math.pi * 100, math.pi / 30)):
    """Function generates timeserise, used for testing"""
    T = [ t for t in np.arange(diapasone[0], diapasone[1], diapasone[2])]
    y_t = np.array([f_Y(a, t) for t in T])
    x_t = np.array([f_X(t) for t in T])
    
    plt.plot(T, y_t)
    plt.plot(T, x_t)
    return (x_t,y_t)