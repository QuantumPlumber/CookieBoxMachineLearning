import importlib
import h5py
import DataHandlingMP as DH

importlib.reload(DH)

DH.transform_2_spectra_from_mp(filename='../AttoStreakSimulations/TF_train_mp_1.hdf5',
                       transfer='reformed_TF_train_mp_1.hdf5')

h5_file = h5py.File('reformed_TF_train_mp_1.hdf5', 'r')
print(h5_file.keys())
for key in list(h5_file.keys()):
    print('shape of {} is {}'.format(key, h5_file[key].shape))

h5_file.close()
