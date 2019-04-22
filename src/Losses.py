import torch
import math
import numpy as np
from src.TSGenerator import f_Y, f_X, dY_dx , get_func_timeseries, f_X_inv, f_X_torch
from src.model_utils import init_logger, plot_shared_scale, plot_multiscale


def aux_loss(mult_par, adit_par, outputs, x_batch, config):
    batch_size = outputs.size()[0]  
    is_debug = config['is_debug']
    is_generated = config['to_generate_data']
    p_gen = config['generator_params']
    a = p_gen['a']
    # f2_arg = f_X_inv(x_batch[-1])
    f2_arg = x_batch[-1]
#     print(f"X shape: {f2_arg.shape}")
#     print(f"Y shape {outputs.shape}")
    f2_arg_k_minus_1 = f2_arg.narrow(0, 1, f2_arg.shape[0] - 1)
    f2_arg_k_minus_2 = f2_arg.narrow(0, 0, f2_arg.shape[0] - 1)
    # f2_arg_k_minus_1 = f2_arg.narrow(0, 1, f2_arg.shape[0] - 1)
#     print(f"f2_arg_k_m1 shape: {f2_arg_k_minus_1.shape}")

    # t_range = np.arange( 0, math.pi * 100, math.pi / 30)
    # x_range = f_X_torch(torch.from_numpy(t_range).type(torch.Tensor))

    # print(f" created X {x_range.shape}")

    dX = x_batch[-1].narrow(0, 1, batch_size -1) - (x_batch[-1].narrow(0, 1, batch_size -1) + x_batch[-1].narrow(0, 0, batch_size-1)) / 2
    f2_mean_drop = (f2_arg_k_minus_1 + f2_arg_k_minus_1) / 2

    dY = dY_dx(a=mult_par, b=adit_par, x=f2_mean_drop) * dX

    y_k_minus_1 = outputs[:, 0].narrow(0, 0, batch_size - 1).view((batch_size - 1, -1))
    
    y_k_hat = y_k_minus_1 + dY
    y_k = outputs[:, 0].narrow(0, 1, batch_size-1).view((batch_size-1, -1))
    error_mul = y_k_hat - y_k
    error_mul = torch.abs(error_mul)
    error_mul = torch.sum(error_mul)

    error_aux = 0
    error_aux += error_mul

    # forsing const:
    if not is_debug:
        const_mul = torch.sum(torch.abs(mult_par.narrow(0, 0, mult_par.shape[0] - 1) - mult_par.narrow(0, 1, mult_par.shape[0] - 1)))
        error_aux += const_mul
        const_adit = torch.sum(torch.abs(adit_par.narrow(0, 0, adit_par.shape[0] - 1) - adit_par.narrow(0, 1, adit_par.shape[0] - 1)))
        error_aux += const_adit

    if is_debug:
        print(y_k_minus_1.shape)
        print(y_k_hat.shape)
        print(dY.shape)
        print(y_k.shape)
        proper_dy = (y_k - y_k_minus_1).view((batch_size - 1, -1))
        print(proper_dy.shape)
        plot_shared_scale([
    #         (f2_arg_k_minus_1.detach().numpy(), "T"),
    #             (y_k_minus_1.detach().numpy(),"Y_k_minus1" ),
    #              (y_k_hat.detach().numpy(), "Y_k"),
            (f2_arg_k_minus_1.detach().numpy()/10, "T"),
                (proper_dy.detach().numpy(), "dY"),
                 (dY.detach().numpy(), "dY_hat")
        ])
   
    return error_aux



def myLoss(outputs, labels, x_batch, config):
    is_debug = config['is_debug']
    is_generated = config['to_generate_data']
    p_gen = config['generator_params']
    a = p_gen['a']
    b = p_gen['b']
    batch_size = outputs.size()[0]  
    mul_par = outputs[:, 1].narrow(0, 1, batch_size - 1).view((batch_size - 1, -1))
    adit_par = outputs[:, 2].narrow(0, 1, batch_size - 1).view((batch_size - 1, -1))
    aux_error = aux_loss(mul_par, adit_par, outputs, x_batch, config)
    if is_debug:
        print(f"Z shape{mul_par.shape}")
        original_a = np.full_like(aux_error.detach().numpy(), a)
        original_a = torch.from_numpy(original_a)
        original_b = np.full_like(aux_error.detach().numpy(), b)
        original_b = torch.from_numpy(original_b)

        x = aux_loss(original_a, original_b, labels, x_batch, config)

        print(f"goal {x}")
        print(f"is {aux_error}")
        while (True):
            pass
#     f2_arg = torch.acos(x_batch[-1])
# #     print(f"X shape: {f2_arg.shape}")
# #     print(f"Y shape {outputs.shape}")
#     #f2_arg_k = f2_arg.narrow(0, 1, f2_arg.shape[0] - 1)
#     f2_arg_k_minus_1 = f2_arg.narrow(0, 0, batch_size - 1)
# #     print(f"f2_arg_k_m1 shape: {f2_arg_k_minus_1.shape}")
#     #d_t = f2_arg_k - f2_arg_k_minus_1
#     z = torch.mean(outputs[:, 1])
# #     z = z.view((z.shape[0], -1))
#     h = math.pi / 30
# #     print(f"z shape{z.shape}")
# #     print(f"prec shape {(z * f2_arg_k_minus_1).shape}")
#     dY = torch.sin(z * f2_arg_k_minus_1) * h
#     print(f"dY shape {dY.shape}")
# #     print(tprch.sum(torch.abs(
# #         (y_k_hat - labels[:, 0].narrow(0,0, batch_size - 1))
#     y_k_minus_1 = outputs.narrow(0, 0, batch_size - 1)
#     y_k_hat = y_k_minus_1 + dY
#     y_k = outputs.narrow(0,1,batch_size-1)
#     error_z = y_k_hat - y_k
#     error_z = torch.abs(error_z)
#     error_z = torch.sum(error_z)
    
    #y_k_minus_1 = outputs.na
#     print(x_batch[-1].detach().numpy().shape)
    
#     print("Out shape")
#     print(outputs.shape)
#     outputs = F.log_softmax(outputs, dim=1)   # compute the log of softmax values
    
#     outputs = torch.abs(outputs - labels)   # compute the log of softmax values
#     print(outputs.shape)
    #outputs = torch.abs(outputs - labels)
    residuals = torch.abs(outputs[:, 0] - labels[:, 0])
    #residuals2 = np.arccos()
    # z = 3
    
#     z = outputs
#     residuals += torch.abs(outputs[:, 1] - 1)
    return torch.sum(residuals) + aux_error
