import numpy as np
import h5py

def get_data_from(data_file_name):
    outfile = h5py.File(data_file_name, 'r')
    print(outfile.keys())
    data = outfile['ans']
    return data

