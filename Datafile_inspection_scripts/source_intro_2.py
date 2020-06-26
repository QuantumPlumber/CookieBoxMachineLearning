import h5py
import numpy as np
import matplotlib.pyplot as plt

# h5_file = h5py.File('../AttoStreakSimulations/TF_train_mp_1.hdf5', 'r')
h5_file = h5py.File('../AttoStreakSimulations/TF_train_widegate_protect.hdf5', 'r')

print(h5_file.keys())
for key in list(h5_file.keys()):
    print('shape of {} is {}'.format(key, h5_file[key].shape))

spectra = h5_file['Spectra'][0:1000, :, :, 1, :]

hist, bins = np.histogram(spectra.flatten(), bins=100)

max = np.max(hist[bins[1:]>5])

plt.figure(figsize=(30, 10))
plt.plot(bins[1:], hist)
plt.grid(b=True, which='minor')
plt.xlim((5,200))
plt.ylim((0,max))

h5_file.close()
