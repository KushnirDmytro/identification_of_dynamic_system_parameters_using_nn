import matplotlib.pyplot as plt
import numpy as np
import math 
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import sklearn.preprocessing 
from src.TSGenerator import f_Y, f_X, dY_dt, f_X_inv, get_func_timeseries

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
        to_generate_data = config['to_generate_data']
        p_data = config['data_params']
        mat_file=p_data['mat_file']
        retrospective_steps=config['network_params']['retrospective_steps']
        need_normalize=p_data['need_normalize']
        leave_nth=p_data['leave_nth']
        if to_generate_data:
            x, y = get_func_timeseries(f_Y = f_Y, f_X = f_X)
        else:
            mat_file = config['data_params']['mat_file']
            outfile = h5py.File(mat_file, 'r')
            #print(outfile.keys())
            self.data  = outfile['ans']
            time, x, y = self.data[::leave_nth, 0], self.data[::leave_nth, 1], self.data[::leave_nth, 2]
        need_normalize = False
        if need_normalize:
            x_normalized, self.x_norms = sklearn.preprocessing.normalize(x.reshape(-1,1),
                                                      axis = 0,
                                                      norm = 'max',
                                                      return_norm = True)
            y_normalized, self.y_norms = sklearn.preprocessing.normalize(y.reshape(-1,1),
                                                      axis = 0,
                                                      norm = 'max', 
                                                      return_norm = True)
            x = x_normalized
            y = y_normalized
#             y = y.reshape(-1,1)
        else:
            x = x.reshape(-1,1)
            y = y.reshape(-1,1)
        print(f"input shape {x.shape}")
        logging.info(f"got data_points: {x.shape}")
        x = x[ int(partition[0] * x.shape[0]) : int(partition[1] * x.shape[0])]
        print(f"input partition shape {x.shape}")
        y = y[ int(partition[0] * y.shape[0]) : int(partition[1] * y.shape[0])]
        print(f"output partition shape {x.shape}")
        x_sliding = []  # determines number of steps for retrospective view
        for i in range(1,retrospective_steps+1):
            x_sliding.append(x[i:-(retrospective_steps+1-i)])
        y = y[retrospective_steps+1:]
        y = np.hstack([y,y])
        print(f"stacked Y shape  {y.shape}")
        self.x = torch.from_numpy(np.array(x_sliding)).type(torch.Tensor)
        self.y = torch.from_numpy(y).type(torch.Tensor)
        print(f"Y_tensor {self.y.shape}")
        print(f"X_tensor {self.x.shape}")
        self.transform = transform

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[:, idx], self.y[idx, :], idx
