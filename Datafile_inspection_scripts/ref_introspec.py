import numpy as np
import h5py
import matplotlib.pyplot as plt

#h5_file = h5py.File('reformed_spectra_densesapce_safe.hdf5', 'r')
#h5_file = h5py.File('reformed_TF_train_mp_1_quarter.hdf5', 'r')
#h5_file = h5py.File('reformed_TF_train_widegate.hdf5', 'r')
h5_file = h5py.File('TF_train_waveform_convert.hdf5', 'r')


print(h5_file.keys())
for key in list(h5_file.keys()):
    print('shape of {} is {}'.format(key, h5_file[key].shape))

spectra_list = []
vn_coeff_list = []
for i in range(100, 110, 1):
    spect = h5_file['Spectra16'][i, :, :]
    spectra_list.append(spect[np.newaxis, ...])
    vn_coeff_list.append(h5_file['VN_coeff'][i, :])

spectra = np.concatenate(spectra_list, axis=0)

fig, ax = plt.subplots(nrows=int(spectra.shape[0]) * 2, ncols=1, figsize=(22, 17), sharex=True)
grid = np.indices(dimensions=(int(spectra.shape[0]), 1))
row = grid[0].flatten() * 2
col = grid[1].flatten()
index = np.arange(spectra.shape[0])
max_y = np.max(spectra)
for ind, ro, co in zip(index, row, col):
    for sp in spectra[ind]:
        ax[ro].plot(sp)

    ax[ro].set_ylim([0, max_y])
    if ind == spectra.shape[0] - 1:
        ax[ro].set_xlabel('electron energy [eV]')

    ax[ro + 1].plot(np.real(vn_coeff_list[ind]))
    ax[ro + 1].plot(np.imag(vn_coeff_list[ind]))

# fig.savefig('Images/KernelDensityEstimate.png', dpi= 700)

h5_file.close()
