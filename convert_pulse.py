import importlib
import h5py
from timeit import default_timer as timer
import time
import DataHandling_pulse as DH

importlib.reload(DH)

if __name__ == '__main__':
    checkpoint_global = time.perf_counter()

    filename = '../AttoStreakSimulations/TF_train_waveform.hdf5'
    transfer = 'convert_test.hdf5'

    DH.transform_2_spectra_from_mp(filename=filename, transfer=transfer)

    delta_t = checkpoint_global - time.perf_counter()
    print('Total Runtime was {}'.format(delta_t))

    h5_file = h5py.File(transfer, 'r')
    print(h5_file.keys())
    for key in list(h5_file.keys()):
        print('shape of {} is {}'.format(key, h5_file[key].shape))
    h5_file.close()
