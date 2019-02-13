import importlib
import h5py
import DataHandling as DH

importlib.reload(DH)

DH.transform_2_spectra(filename='../AttoStreakSimulations/TF_training_densespace.hdf5',
                       transfer='reformed_spectra_densesapce.hdf5')

h5_file = h5py.File('reformed_spectra_densesapce.hdf5','r')
print(h5_file.keys())
for key in list(h5_file.keys()):
    print('shape of {} is {}'.format(key, h5_file[key].shape))

h5_file.close()
