h5_file = h5py.File('reformed_TF_train_mp_1.hdf5', 'r')
print(h5_file.keys())
for key in list(h5_file.keys()):
    print('shape of {} is {}'.format(key, h5_file[key].shape))

grab_spect = np.arange(0, 100, 10)
spectra = h5_file['VN_coeff'][grab_spect, :]

fig, ax = plt.subplots(nrows=int(spectra.shape[0]), ncols=1, figsize=(22, 17), sharex=True)
grid = np.indices(dimensions=(int(spectra.shape[0]), 1))
row = grid[0].flatten()
col = grid[1].flatten()
index = np.arange(spectra.shape[0])
max_y = np.max(spectra)
for ind, ro, co in zip(index, row, col):
    ax[ro].plot(spectra[ind], 'b')
    ax[ro].set_ylim([0, max_y])
    if ind == spectra.shape[0] - 1:
        ax[ro].set_xlabel('electron energy [eV]')

# fig.savefig('Images/KernelDensityEstimate.png', dpi= 700)

h5_file.close()
