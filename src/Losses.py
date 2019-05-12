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


def core_loss(outputs, labels, config):

    dy_observed = (labels[1:] - labels[:-1]).abs() # used here exclusively for weights

    # only_active_dy = torch.where(dy_observed > config['train_params']['aux']['steady_state_threshold'],
    #                              dy_observed,
    #                              torch.zeros_like(dy_observed))

    # active_indexes = only_active_dy.nonzero()[:, 0]
    # nonzero_mean = dy_observed[active_indexes].mean()
    # nonzero_max = dy_observed[active_indexes].mean()

    stabilizing_part = config['train_params']['core_loss']['part_of_max_for_const']
    mx = dy_observed.abs().max() * stabilizing_part
    w = dy_observed + mx  # use some weights to balance steady state error

    residuals = ((outputs - labels)[:-1] * w).unsqueeze(1)
    # residuals = (residuals * w)#.unsqueeze(dim=1)
    # r = residuals.abs().sum()

    scaling_order = config['train_params']['aux_loss']['scaling_order']
    r = residuals.t().mm(residuals) / residuals.shape[0] * 10**scaling_order


    # residuals = (residuals * w).unsqueeze(dim=1)
    #
    # dynamic_error = residuals * dy_observed.unsqueeze(1)
    # print(dynamic_error.sum())
    # print(dynamic_error.shape)
    # stable_state_error = residuals * dy_observed.mean()
    # print(f"stable {stable_state_error.sum() }")
    # print(f"stable {stable_state_error.shape }")
    # r = dynamic_error + stable_state_error
    # r = r.t().mm(r) / r.shape[0]
    # loss = residuals.t().mm(residuals) / residuals.shape[0] # representing squared and reduce sum simultaniusly
    # print(f"{r.abs().sum()} ?= {residuals.abs().sum()}")

    return r #+ r.log()


def aux_loss(inputs, outputs, params, labels, config):

    par_1, par_2 = params
    par_1_mean = torch.mean(par_1)
    par_2_mean = torch.mean(par_2)

    eps = 0.01

    if par_1_mean > 0 + eps and par_2_mean > 0 + eps:

        y_k_m1 = outputs[:-1]

        v_a = inputs[-1][1:]

        sparse_data_step = config['data_params']['leave_nth']
        integreation_timestep = config['data_params']['integration_step']
        d_t = integreation_timestep * (sparse_data_step - 2)  # const 2 used manually

        parameters = {
            'y_k_m1': y_k_m1,  # time series previous steps
            'Kt': 0.001,  # motor torque constant
            'Kb': 0.01,  # emf constant
            'v_a': v_a,  # voltage, governing signal
            'La': par_1.mean(),  # armature resistance
            'Ra': par_2.mean(),  # armature inductiveness
            # if use not means but whole timeseries instead, we'll have a problem, because model will try to fit
            # parameter value for piece of train data and creating a reliable const loss will be a challanging task
            'd_t': d_t  # timestamp difference, aka integration step
        }

        dy_predicted = prognose_dy(parameters)

        dy_observed = labels[1:] - labels[:-1]
        # from matplotlib import pyplot as plt

        weights = dy_observed.abs()
        weights -= weights.min()
        weights /= (weights.max() - weights.min())
        stabilizing_part = config['train_params']['aux_loss']['part_of_max_for_const']
        mx = weights.max() * stabilizing_part
        weights += mx
        # using normalisation and standartisation to obtain better convergance
        # along values with marginal 0 value of dy

        aux_residuals = dy_predicted - dy_observed
        aux_residuals *= weights  # using dy_observed as weighting coefficients
        # reduced_error = aux_residuals.t().mm(aux_residuals) # / aux_residuals.shape[0]
        # todo provide loss select to unify
        reduced_error = aux_residuals.abs().sum()

        aux_error = reduced_error #torch.log(reduced_error)
        # idea is to make this loss on order of magnitude higher then usual loss

    else:
        aux_error = torch.abs(torch.min(par_1_mean,
                                         par_2_mean) - eps)
        # because those parameters are strictly positive physical parameters
    return aux_error

def aux_loss_jordan(inputs, outputs, params, labels, config):

    par_1, par_2 = params
    par_1_mean = par_1
    par_2_mean = par_2

    eps = 0.01
    y_k_m1 = outputs[:-1]

    v_a = inputs[-1][1:]

    sparse_data_step = config['data_params']['leave_nth']
    integreation_timestep = config['data_params']['integration_step']
    d_t = integreation_timestep * (sparse_data_step - 2)  # const 2 used manually

    parameters = {
        'y_k_m1': y_k_m1,  # time series previous steps
        'Kt': 0.001,  # motor torque constant
        'Kb': 0.01,  # emf constant
        'v_a': v_a,  # voltage, governing signal
        'La': par_1,  # armature resistance
        'Ra': par_2,  # armature inductiveness
        # if use not means but whole timeseries instead, we'll have a problem, because model will try to fit
        # parameter value for piece of train data and creating a reliable const loss will be a challanging task
        'd_t': d_t  # timestamp difference, aka integration step
    }

    dy_predicted = prognose_dy(parameters)

    dy_observed = labels[1:] - labels[:-1]
    # from matplotlib import pyplot as plt

    weights = dy_observed.abs()
    weights -= weights.min()
    weights /= (weights.max() - weights.min())
    stabilizing_part = config['train_params']['aux_loss']['part_of_max_for_const']
    mx = weights.max() * stabilizing_part
    weights += mx
    # using normalisation to obtain better convergance
    # along values with marginal 0 value of dy

    aux_residuals = dy_predicted - dy_observed
    aux_residuals *= weights
    if par_1_mean > 0 + eps and par_2_mean > 0 + eps:

        # using dy_observed as weighting coefficients
        # reduced_error = aux_residuals.t().mm(aux_residuals) # / aux_residuals.shape[0]
        reduced_error = aux_residuals.abs().sum() + aux_residuals.t().mm(aux_residuals)


        aux_error = reduced_error  #torch.log(reduced_error)
        # idea is to make this loss on order of magnitude higher then usual loss

    else:
        aux_error = torch.abs(torch.min(par_1_mean, par_2_mean) - eps)
        print(f'Negative param value penalty {par_1_mean} {par_2_mean}')
        # because those parameters are strictly positive physical parameters
    return aux_error, aux_residuals


def const_param_loss(pars):
    const_loss = torch.zeros(len(pars)) # explicit storage done for debugging and visualising purposes
    for i, par in enumerate(pars):
        # c_loss = par.var() + par.std()
        c_loss = par - par.mean()
        c_loss = c_loss.t().mm(c_loss) #/ c_loss.shape[0]
        const_loss[i] = c_loss # + c_loss.log()
    return const_loss.sum()

def myLoss_jordan(outputs, jordan, labels, x_batch, config):

    x_norm = config['x_norm']
    y_norm = config['y_norm']

    outputs_denorm = outputs * y_norm
    labels_denorm = labels * y_norm
    x_batch_denorm = x_batch * x_norm

    par_1 = jordan[0]
    par_2 = jordan[1]

    E = core_loss(outputs=outputs[:, 0],
                  labels=labels[:, 0],
                  config=config)  # this loss computed on normalized data

    aux, aux_residuals = aux_loss_jordan(inputs=x_batch_denorm,
                outputs=outputs_denorm[:, 0].unsqueeze(dim=1),
                params=[par_1, par_2],
                labels=labels_denorm[:, 0].unsqueeze(dim=1),
                config=config)

    jordan_loss, jordan_aux_residuals = aux_loss_jordan(inputs=x_batch_denorm,
                                         outputs=labels_denorm[:, 0].unsqueeze(dim=1),
                                         params=[par_1, par_2],
                                         labels=labels_denorm[:, 0].unsqueeze(dim=1),
                                         config=config)



    return E, aux, aux_residuals, jordan_loss

def myLoss(outputs, labels, x_batch, config):

    x_norm = config['x_norm']
    y_norm = config['y_norm']

    outputs_denorm = outputs * y_norm
    labels_denorm = labels * y_norm
    x_batch_denorm = x_batch * x_norm

    par_1 = outputs[:-1, 1].unsqueeze(dim=1)  # no sense to renormalize those
    par_2 = outputs[:-1, 2].unsqueeze(dim=1)

    E = core_loss(outputs=outputs[:, 0],
                  labels=labels[:,0],
                  config=config)  # this loss computed on normalized data

    aux = aux_loss(inputs=x_batch_denorm,
                outputs=outputs_denorm[:, 0].unsqueeze(dim=1),
                params=[par_1, par_2],
                labels=labels_denorm[:, 0].unsqueeze(dim=1),
                config=config)

    const_loss = const_param_loss(pars = [par_1, par_2])

    return E , aux, const_loss
