import matplotlib.pyplot as plt
import numpy as np
import math 
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import sklearn.preprocessing 
from src.TSGenerator import f_Y, f_X, dY_dx, get_func_timeseries


class TimeSeriesDataset(Dataset):
    """Loads dataset from matlab file and provides batching interface"""

    def __init__(self, 
                 config, 
                 partition,
                 logging,
                 transform=None):
        """
        Args:
            config (dict): configurations dict
            partition (list): indexes to select 
            logging (Object): custom logger
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        
        # TODO asserts and validations of provided params
        self.to_generate_data = config['to_generate_data']
        self.p_data = config['data_params']
        self.mat_file = self.p_data['mat_file']
        self.p_net = config['network_params']
        self.retrospective_steps = self.p_net['retrospective_steps']
        self.out_dim = self.p_net['output_dim']
        self.need_normalize = self.p_data['need_normalize']
        self.leave_nth = self.p_data['leave_nth']
        if self.to_generate_data:
            p_generate = config['generator_params']
            a = p_generate['a']
            b = p_generate['b']
            self.x, self.y = get_func_timeseries(f_Y=f_Y, f_X=f_X, a=a, b=b)
        else:
            outfile = h5py.File(self.mat_file, 'r')
            self.data = outfile['ans']
            time, self.x, self.y = self.data[::self.leave_nth, 0], self.data[::self.leave_nth, 1], self.data[::self.leave_nth, 2]
            self.x = self.x.reshape(-1, 1)
            self.y = self.y.reshape(-1, 1)
        if self.need_normalize:


            x_normalized, self.x_norms = sklearn.preprocessing.normalize(self.x,
                                                                         axis=0,
                                                                         norm='max',
                                                                         return_norm=True)
            print(f"X normalized, norms: {self.x_norms}")
            print(f"x shape {self.x.shape}")
            print(f"x_normalized shaoe {x_normalized.shape}")
            plt.plot(self.x)
            plt.title('x original')
            plt.show()
            plt.plot(x_normalized)
            plt.title('normalized x')
            plt.show()
            y_normalized, self.y_norms = sklearn.preprocessing.normalize(self.y,
                                                                         axis=0,
                                                                         norm='max',
                                                                         return_norm=True)

            print(f"Y normalized, norms: {self.y_norms}")

            x = x_normalized
            y = y_normalized
#             y = y.reshape(-1,1)
#         else:
#             x = self.x.reshape(-1,1)
#             y = self.y.reshape(-1,1)
        print(f"input shape {x.shape}")
        logging.info(f"got data_points: {x.shape}")
        x = x[ int(partition[0] * x.shape[0]) : int(partition[1] * x.shape[0])]
        print(f"input partition shape {x.shape}")
        y = y[ int(partition[0] * y.shape[0]) : int(partition[1] * y.shape[0])]
        print(f"output partition shape {x.shape}")

        x_sliding = []  # determines number of steps for retrospective view
        for i in range(1, self.retrospective_steps+1):
            x_sliding.append(x[i:-(self.retrospective_steps+1-i)])
        y = y[self.retrospective_steps-1:]

        print(f"stacked Y shape  {y.shape}")

        self.x = torch.as_tensor(x_sliding).type(torch.Tensor)
        self.y = torch.as_tensor(y).type(torch.Tensor)

        print(f"Y_tensor {self.y.shape}")
        print(f"X_tensor {self.x.shape}")

        self.transform = transform

    def build_x_retrospectice(self, idx):
        x_sliding = []  # determines number of steps for retrospective view
        for i in range(idx, idx + self.retrospective_steps + 1):
            x_sliding.append(self.x[i:i+self.retrospective_steps + 1])
        return torch.as_tensor(x_sliding).type(torch.Tensor)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[:, idx], self.y[idx, :], idx
