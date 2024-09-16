import h5py
import numpy as np

def save_to_hdf5(file_name, data, labels):
    with h5py.File(file_name, 'w') as f:
        f.create_dataset('data', data=data)
        f.create_dataset('labels', data=labels)

def load_from_hdf5(file_name):
    with h5py.File(file_name, 'r') as f:
        data = f['data'][:]
        labels = f['labels'][:]
    return data, labels
