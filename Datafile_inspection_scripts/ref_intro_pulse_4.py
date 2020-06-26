import numpy as np
import h5py
import matplotlib.pyplot as plt

h5_file = h5py.File('Data/25_hit_5-14-19/25_hit_5-14-19_convert.hdf5', 'r')


print(h5_file.keys())
for key in list(h5_file.keys()):
    print('shape of {} is {}'.format(key, h5_file[key].shape))

select = np.random.random_integers(0, 100000, 1)
print(select)
spectra = h5_file['Spectra16'][select[0], :, :]

h5_file.close()

fig, ax = plt.subplots(nrows=int(spectra.shape[0]), ncols=1, figsize=(22, 17), sharex=True)
grid = np.indices(dimensions=(int(spectra.shape[0]), 1))
row = grid[0].flatten()
col = grid[1].flatten()
index = np.arange(spectra.shape[0])
max_y = np.max(spectra)
for ind, ro, co in zip(index, row, col):
    print(ind)
    print(ro)
    ax[ro].plot(spectra[ind])

    ax[ro].set_ylim([0, max_y])
    if ind == spectra.shape[0] - 1:
        ax[ro].set_xlabel('electron energy [eV]')


# fig.savefig('Images/KernelDensityEstimate.png', dpi= 700)

