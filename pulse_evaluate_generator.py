import importlib
import h5py
from timeit import default_timer as timer
import time
import DataHandling_pulse as DH

'''
A function to combine training data pulses for testing

'''

importlib.reload(DH)

if __name__ == '__main__':
    checkpoint_global = time.perf_counter()

    filename = 'TF_train_wave_unwrapped.hdf5'
    transfer = 'TF_train_wave_unwrapped_eggs.hdf5'

    DH.pulse_evaluate(filename=filename, transfer=transfer, num_spectra=int(1e6))

    delta_t = checkpoint_global - time.perf_counter()
    print('Total Runtime was {}'.format(delta_t))

    h5_file = h5py.File(transfer, 'r')
    print(h5_file.keys())
    for key in list(h5_file.keys()):
        print('shape of {} is {}'.format(key, h5_file[key].shape))
    h5_file.close()