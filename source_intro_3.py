import h5py
import numpy as np
import matplotlib.pyplot as plt

# h5_file = h5py.File('../AttoStreakSimulations/TF_train_mp_1.hdf5', 'r')
h5_file = h5py.File('../AttoStreakSimulations/TF_train_widegate_protect.hdf5', 'r')

print(h5_file.keys())
for key in list(h5_file.keys()):
    print('shape of {} is {}'.format(key, h5_file[key].shape))

vn_coeff = h5_file['VN_coeff'][0:500, ...]
vn_coeff_reshape = np.reshape(vn_coeff, newshape=(vn_coeff.shape[0], vn_coeff.shape[1] * vn_coeff.shape[2]))

maxxes = np.amax(np.abs(vn_coeff_reshape), axis=1)

hist, bins = np.histogram(maxxes, bins=100)

plt.figure(figsize=(30, 10))
plt.plot(bins[1:], hist)
plt.grid()


vn_coeff_real = vn_coeff.real.flatten()
vn_coeff_imag = vn_coeff.imag.flatten()

real_hist, real_bins = np.histogram(vn_coeff_real, bins=300)
imag_hist, imag_bins = np.histogram(vn_coeff_imag, bins=300)


plt.figure(figsize=(30, 10))
plt.plot(real_bins[1:], real_hist)
plt.plot(imag_bins[1:], imag_hist)
plt.grid()


h5_file.close()
