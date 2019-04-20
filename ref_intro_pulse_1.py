import h5py
import numpy as np
import matplotlib.pyplot as plt

#h5_file = h5py.File('convert_test.hdf5', 'r')
#h5_file = h5py.File('TF_train_wave_unwrapped.hdf5', 'r')
#h5_file = h5py.File('../Data/unwrapped/Eggs/TF_train_wave_unwrapped_eggs.hdf5', 'r')
#h5_file = h5py.File('/Data/25Hit_unwrapped/Scrambled/TF_train_25Hits_eggs.hdf5', 'r')
h5_file = h5py.File('Data/unwrapped_step/TF_train_waveform_unwrapped_step.hdf5', 'r')


print(h5_file.keys())
for key in list(h5_file.keys()):
    print('shape of {} is {}'.format(key, h5_file[key].shape))

select = np.sort(np.random.random_integers(0, 100000, 10))
print(select)
# select = np.arange(5)
pulse_pure = h5_file['Pulse_truth'][select, :, :]

h5_file.close()

pulse = pulse_pure.copy()
#pulse[:, 1, 1:] -= pulse[:, 1, :-1]
pulse_recon = pulse[:, :1, :] * np.exp(-1J * pulse[:, 1:, :])

#pulse_total = np.concatenate((pulse_pure, pulse), axis=1)
pulse_total = np.concatenate((pulse, np.abs(pulse_recon), pulse_recon.real, pulse_recon.imag), axis=1)


fig, ax = plt.subplots(nrows=int(pulse_total.shape[0]), ncols=int(pulse_total.shape[1]), figsize=(22, 17),
                       sharex=True)
grid = np.indices(dimensions=(int(pulse_total.shape[0]), int(pulse_total.shape[1])))
row = grid[0].flatten()
col = grid[1].flatten()
index = np.arange(pulse.shape[0])

for ro, co in zip(row, col):
    ax[ro, co].plot(pulse_total[ro, co, :], '-')

# fig.savefig('Images/KernelDensityEstimate.png', dpi=700)
