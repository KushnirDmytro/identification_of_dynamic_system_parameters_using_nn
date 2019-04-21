import torch
import math
import numpy as np
from src.TSGenerator import f_Y, f_X, dY_dt, f_X_inv, get_func_timeseries
from src.model_utils import init_logger, plot_shared_scale, plot_multiscale


def aux_loss(z, outputs, x_batch, config):
    batch_size = outputs.size()[0]  
    is_debug = config['is_debug']
    f2_arg = f_X_inv(x_batch[-1])
    
#     print(f"X shape: {f2_arg.shape}")
#     print(f"Y shape {outputs.shape}")
    f2_arg_k_minus_1 = f2_arg.narrow(0, 1, f2_arg.shape[0] - 1)
#     print(f"f2_arg_k_m1 shape: {f2_arg_k_minus_1.shape}")
    #d_t = f2_arg_k - f2_arg_k_minus_1
    #z = torch.mean()
#     z = z.narrow(0, 0, batch_size - 1).view((z.shape[0], -1))
    h = math.pi / 30
#     print(f"z shape{z.shape}")
#     print(f"prec shape {(z * f2_arg_k_minus_1).shape}")
    dY = dY_dt(z, f2_arg_k_minus_1) * h
#     dY = torch.cos(z * f2_arg_k) * h
    
#     print(f"dY shape {dY.shape}")
#     print(tprch.sum(torch.abs(
#         (y_k_hat - labels[:, 0].narrow(0,0, batch_size - 1))

    
    y_k_minus_1 = outputs[:,0].narrow(0, 0, batch_size - 1).view((batch_size - 1, -1))
    
    y_k_hat = y_k_minus_1 + dY
    y_k = outputs[:,0].narrow(0,1,batch_size-1).view((batch_size-1, -1))
    error_z = y_k_hat - y_k
    error_z = torch.abs(error_z)
    error_z = torch.sum(error_z)
    # forsing const:
    if not is_debug:
        const = torch.sum(torch.abs(z.narrow(0,0,z.shape[0]-1) - z.narrow(0, 1, z.shape[0] - 1)))
        error_z += const
    
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
   
    return error_z

def myLoss(outputs, labels, x_batch, config):
    is_debug = config['is_debug']
        
    batch_size = outputs.size()[0]  
    z = outputs[:, 1].narrow(0,1,batch_size - 1).view((batch_size - 1, -1))
    z = aux_loss(z, outputs, x_batch, config)
    if is_debug:
        print(f"Z shape{z.shape}")
        a = config['debug_params']['a']
        original_z = np.full_like(z.detach().numpy(), a)
        original_z = torch.from_numpy(original_z)
        x = aux_loss(original_z, labels, x_batch, config)

        print(f"goal {x}")
        print(f"is {z}")
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
    return torch.sum(residuals) + z