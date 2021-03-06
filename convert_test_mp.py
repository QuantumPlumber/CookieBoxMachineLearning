import importlib
import h5py
from timeit import default_timer as timer
import time
import DataHandlingMP as DH

importlib.reload(DH)

if __name__ == '__main__':
    checkpoint_global = time.perf_counter()

    # filename = '../AttoStreakSimulations/TF_train_mp_1.hdf5'
    # transfer = 'reformed_TF_train_mp_1_quarter_stop.hdf5'

    #filename = '../AttoStreakSimulations/TF_train_widegate_protect.hdf5'
    #transfer = 'reformed_TF_train_widegate.hdf5'

    filename = '../AttoStreakSimulations/TF_train_update.hdf5'
    transfer = 'convert_test.hdf5'

    DH.transform_2_spectra_from_mp(filename=filename, transfer=transfer)

    delta_t = checkpoint_global - time.perf_counter()
    print('Total Runtime was {}'.format(delta_t))

    h5_file = h5py.File(transfer, 'r')
    print(h5_file.keys())
    for key in list(h5_file.keys()):
        print('shape of {} is {}'.format(key, h5_file[key].shape))

    h5_file.close()
