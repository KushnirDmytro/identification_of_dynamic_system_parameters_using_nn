import torch
import math
import numpy as np
from src.TSGenerator import f_Y, f_X, dY_dx , get_func_timeseries, f_X_inv, f_X_torch
from src.model_utils import init_logger, plot_shared_scale, plot_multiscale


def prognose_dy(par):
    y_k_m1 = par['y_k_m1']
    v_a = par['v_a']
    La = par['La']
    Ra = par['Ra']
    Kt = par['Kt']
    Kb = par['Kt']
    d_t = par['d_t']

    i_k_m1 = y_k_m1 / Kt  # restoring estimates from observable state y_k_m1

    e_k_m1 = i_k_m1 * Kt * Kb  # EMF

    di_dt = (v_a - e_k_m1 - i_k_m1 * Ra)/La  # current change

    i_k = i_k_m1 + di_dt * d_t  # resulting estimate of current
    y_k = i_k * Kt  # resulting  torque
    d_y = y_k - y_k_m1

    return d_y


def core_loss(outputs, labels):
    dy_observed = labels[1:] - labels[0:-1]  # used here exclusively for weights

    residuals = (outputs - labels)[:-1]
    residuals = (residuals * 0.001 + residuals * dy_observed).unsqueeze(dim=1)
    loss = residuals.t().mm(residuals)  # representing squeare and reduce sum simultaniusly
    return loss

def aux_loss():
    pass

def const_param_loss(pars):
    const_loss = np.zeros(len(pars)) #  explicit storage done for debugging and visualising purposes
    for i, par in enumerate(pars):
        a = par.var()
        const_loss[i] = par.var()
    return np.sum(const_loss)

def myLoss(outputs, labels, x_batch, config):

    x_norm = config['x_norm']
    y_norm = config['y_norm']

    outputs_denorm = outputs * y_norm
    labels_denorm = labels * y_norm
    x_batch_denorm = x_batch * x_norm

    batch_size = outputs.size()[0]


    E = core_loss(outputs=outputs_denorm[:, 0], labels=labels_denorm[:,0])

#       AUX _ PART

    # todo replace with cumdiff
    # todo make better unpacking way for parameters
    y_k_m1 = outputs_denorm[:, 0].narrow(0, 0, batch_size - 1).view((batch_size - 1, -1))

    v_a = x_batch_denorm[-1].narrow(0, 1, batch_size - 1)
    par_1 = outputs[:, 1].narrow(0,0,batch_size-1).view((batch_size-1, -1))
    par_2 = outputs[:, 2].narrow(0,0,batch_size-1).view((batch_size-1, -1))


    sparse_data_step = config['data_params']['leave_nth']
    d_t = 0.01 * (sparse_data_step-2)  # todo check optimal  # sampling time times droprate

    parameters = {
        'y_k_m1': y_k_m1,  # time series previous steps
        'Kt': 0.001,  # motor torque constant
        'Kb': 0.01,  # emf constant
        'v_a': v_a,  # voltage, governing signal
        'La': par_1,  # armature resistance
        'Ra': par_2,  # armature inductiveness
        'd_t': d_t  # timestamp difference, aka integration step
    }

    dy_predicted = prognose_dy(parameters)

    error_aux = 0
    const_loss = 0
    dy_observed = labels_denorm[:, 0].narrow(0, 1, batch_size - 1) - labels_denorm[:, 0].narrow(0, 0, batch_size - 1)

    dy_observed = dy_observed.view((batch_size - 1, -1))
    aux_residuals = dy_predicted - dy_observed
    aux_residuals *= dy_observed  # using them as weighting coefficients

    reduced_error = aux_residuals.t().mm(aux_residuals)

    aux_error = reduced_error + torch.log(reduced_error)

    par_1_mean = torch.mean(par_1)
    par_2_mean = torch.mean(par_2)
    eps = 0.01

    if par_1_mean > 0 + eps and par_2_mean > 0 + eps:
        # todo check improvenents for it
        # error_aux += torch.log(aux_error)  # in order to increase curvature of loss fn
        error_aux += aux_error
    else:
        error_aux += torch.abs(torch.min(par_1_mean, par_2_mean) - eps)  # because those parameters are strictly positive phisical parameters

    const_loss = const_param_loss(pars = [par_1, par_2])

#     /  AUX _ PART


    return E, error_aux, const_loss
