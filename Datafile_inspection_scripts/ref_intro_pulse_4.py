import numpy as np
import h5py
import matplotlib.pyplot as plt

# h5_file = h5py.File('Data/25_hit_5-14-19/25_hit_5-14-19_convert.hdf5', 'r')
h5_file = h5py.File('../../AttoStreakSimulations/Data/TF_another_set.hdf5', 'r')

print(h5_file.keys())
for key in list(h5_file.keys()):
    print('shape of {} is {}'.format(key, h5_file[key].shape))

source_shape = h5_file['Spectra'].shape
select = np.random.random_integers(0, source_shape[0], 2)
print(select)
spectra = h5_file['Spectra'][select[0], :, :, :]
time_waveform = h5_file['Time_pulse'][select[0], :]
phase_pulse = h5_file['Phase_pulse'][select[0], :]
nonlinearphase_pulse = h5_file['Nonlinearphase_pulse'][select[0], :]

h5_file.close()

fig, ax = plt.subplots(nrows=int(spectra.shape[0]) + 4, ncols=1, figsize=(22, 17), sharex=True)
# plt.tight_layout()
grid = np.indices(dimensions=(int(ax.shape[0]), 1))
row = grid[0].flatten()
col = grid[1].flatten()
index = np.arange(ax.shape[0])
print(index)
real_spectra = np.sum(spectra ** 2, axis=2)*1e9
#real_spectra = real_spectra / (np.max(real_spectra))
max_y = 1.1*np.max(real_spectra)
min_y = 1.1*np.min(real_spectra)
#real_spectra = real_spectra / max_y
for ind, ro, co in zip(index, row, col):
    print(ind)
    # print(ro)
    if ind < index[-4]:
        #ax[ro].plot(spectra[ind, :, 0])
        #ax[ro].plot(spectra[ind, :, 1])
        ax[ro].plot(real_spectra[ind, :])
        ax[ro].plot(np.ones_like(real_spectra[ind, :]))

        #ax[ro].set_ylim([min_y, max_y])

    #if ind == index[-4]:
        #ax[ro].set_xlabel('electron energy [eV]')

    if ind == index[-3]:
        ax[ro].plot(time_waveform)
        #ax[ro].set_xlabel('time [fs]')
        #ax[ro].set_ylim([0, 1.1*np.max(time_waveform)])

    if ind == index[-2]:
        ax[ro].plot(phase_pulse)
        #ax[ro].set_xlabel('time [fs]')
        #ax[ro].set_ylim([1.1*np.min(phase_pulse), 1.1*np.max(phase_pulse)])

    if ind == index[-1]:
        ax[ro].plot(nonlinearphase_pulse)
        #ax[ro].set_xlabel('time [fs]')
        #ax[ro].set_ylim([1.1*np.min(nonlinearphase_pulse), 1.1*np.max(nonlinearphase_pulse)])


# fig.savefig('Images/KernelDensityEstimate.png', dpi= 700)
