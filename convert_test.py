import importlib
import h5py
import DataHandling as DH

importlib.reload(DH)

DH.transform_2_spectra(filename='../AttoStreakSimulations/TF_train_single.hdf5', transfer='reformed_spectra.hdf5')

#h5_file = h5py.File('reformed_spectra.hdf5','r')
#print(h5_file.keys())
