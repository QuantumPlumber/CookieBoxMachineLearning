import numpy as np
import matplotlib.pyplot as plt
import h5py

h5_file = h5py.File('reformed_spectra_densesapce_safe.hdf5', 'r')
# h5_file = h5py.File('reformed_TF_train_mp_1_quarter.hdf5', 'r')
# h5_file = h5py.File('reformed_TF_train_widegate.hdf5', 'r')

print(h5_file.keys())
for key in list(h5_file.keys()):
    print('shape of {} is {}'.format(key, h5_file[key].shape))

vn_coeff = h5_file['VN_coeff'][:, :]

#un_normed = np.where(np.abs(vn_coeff) > 1.1)
#print(un_normed)

max_hist, max_bins = np.histogram(np.amax(np.abs(vn_coeff), axis=1), bins=300)
plt.figure(figsize=(30, 10))
plt.plot(max_bins[1:], max_hist)
plt.grid()

'''
real_hist, real_bins = np.histogram(vn_coeff.flatten().real, bins=300)
imag_hist, imag_bins = np.histogram(vn_coeff.flatten().imag, bins=300)


plt.figure(figsize=(30, 10))
plt.plot(real_bins[1:], real_hist)
plt.plot(imag_bins[1:], imag_hist)
plt.grid()
'''
h5_file.close()
