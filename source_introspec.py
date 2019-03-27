import h5py
import numpy as np
import matplotlib.pyplot as plt

#h5_file = h5py.File('../AttoStreakSimulations/TF_train_mp_1.hdf5', 'r')
h5_file = h5py.File('../AttoStreakSimulations/TF_train_widegate_protect.hdf5', 'r')

print(h5_file.keys())
for key in list(h5_file.keys()):
    print('shape of {} is {}'.format(key, h5_file[key].shape))

hits = h5_file['Hits'][...]

hist, bins = np.histogram(hits.flatten(), bins=int(hits.max() - hits.min()))

plt.figure(figsize=(30, 10))
plt.plot(bins[1:], hist)

hits_sum = np.sum(hits, axis=2)
hist_sum, bins_sum = np.histogram(hits_sum.flatten(), bins=int(hits_sum.max() - hits_sum.min()))

plt.figure(figsize=(30, 10))
plt.plot(bins_sum[1:], hist_sum)

hits_full = np.reshape(hits, newshape=(hits.shape[0] * hits.shape[1], hits.shape[2]))
test_full = np.any(hits_full == 199, axis=1)
hist_full, bins_full = np.histogram(test_full)

plt.figure(figsize=(30, 10))
plt.plot(bins_full[1:], hist_full)

h5_file.close()
