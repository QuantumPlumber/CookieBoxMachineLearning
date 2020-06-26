import numpy as np
import matplotlib.pyplot as plt
import h5py

h5_file_old = h5py.File('reformed_spectra_densesapce_safe.hdf5', 'r')
# h5_file = h5py.File('reformed_TF_train_mp_1_quarter.hdf5', 'r')
h5_file_new = h5py.File('reformed_TF_train_widegate.hdf5', 'r')

print(h5_file_old.keys())
for key in list(h5_file_old.keys()):
    print('shape of {} is {}'.format(key, h5_file_old[key].shape))

actual_num = 6
old_num = actual_num * 3
new_num = actual_num * 10
spectra_old = h5_file_old['Spectra16'][old_num, :, :]
spectra_new = h5_file_new['Spectra16'][new_num, :, :]
vn_old = h5_file_old['VN_coeff'][old_num, :]
vn_new = h5_file_new['VN_coeff'][new_num, :]

print(spectra_new == spectra_old)

fig, ax = plt.subplots(nrows=int(spectra_new.shape[0]), ncols=1, figsize=(22, 17), sharex=True)
grid = np.indices(dimensions=(int(spectra_new.shape[0]), 1))
row = grid[0].flatten()
col = grid[1].flatten()
index = np.arange(spectra_new.shape[0])
max_y = np.max(spectra_new)
for ind, ro, co in zip(index, row, col):
    ax[ro].plot(spectra_old[ind], 'r')
    ax[ro].plot(spectra_new[ind], 'b')

# vn_old = vn_old / np.abs(vn_old).max()

fig2, ax2 = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=True)
ax2[0, 0].pcolormesh(np.reshape(np.abs(vn_old), newshape=(10, 10)))
ax2[0, 1].pcolormesh(np.reshape(np.angle(vn_old), newshape=(10, 10)))

ax2[1, 0].pcolormesh(np.reshape(np.abs(vn_new), newshape=(10, 10)))
ax2[1, 1].pcolormesh(np.reshape(np.angle(vn_new), newshape=(10, 10)))


h5_file_old.close()
h5_file_new.close()
