import numpy as np
import matplotlib.pyplot as plt
import h5py

#h5_file = h5py.File('reformed_spectra_densesapce_safe.hdf5', 'r')
#h5_file = h5py.File('reformed_TF_train_mp_1_quarter.hdf5', 'r')
#h5_file = h5py.File('reformed_TF_train_widegate.hdf5', 'r')
h5_file = h5py.File('../Data/convert/TF_train_waveform_convert.hdf5', 'r')

print(h5_file.keys())
for key in list(h5_file.keys()):
    print('shape of {} is {}'.format(key, h5_file[key].shape))

grab_spect = np.arange(306, 312, 1)
spectra = h5_file['Pulse_truth'][grab_spect, 0, :]
#spectra = spectra/np.max(np.abs(spectra))

fig, ax = plt.subplots(nrows=int(spectra.shape[0]), ncols=1, figsize=(22, 17), sharex=True)
grid = np.indices(dimensions=(int(spectra.shape[0]), 1))
row = grid[0].flatten()
col = grid[1].flatten()
index = np.arange(spectra.shape[0])
max_y = np.max(np.abs(spectra))
min_y = np.min(spectra)
for ind, ro, co in zip(index, row, col):
    ax[ro].plot(spectra[ind].real, '.b')
    ax[ro].plot(spectra[ind].imag, '.r')
    ax[ro].set_ylim([-max_y, max_y])
    if ind == spectra.shape[0] - 1:
        ax[ro].set_xlabel('electron energy [eV]')

# fig.savefig('Images/KernelDensityEstimate.png', dpi= 700)

h5_file.close()
