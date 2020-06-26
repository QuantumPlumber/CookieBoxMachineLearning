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
vn_new = h5_file_new['VN_coeff'][0:1000, :]

results_uniq, results_count = np.unique(vn_new, return_counts=True, axis=0)

count_hist, count_bins = np.histogram(results_count, bins=300)
plt.figure(figsize=(30, 10))
plt.plot(count_bins[1:], count_hist)
plt.grid()



h5_file_old.close()
h5_file_new.close()

