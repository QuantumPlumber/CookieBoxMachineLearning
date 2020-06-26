import h5py
import numpy as np
import matplotlib.pyplot as plt

filename = '../AttoStreakSimulations/TF_train_waveform.hdf5'
h5_file = h5py.File(filename, 'r')
print(h5_file.keys())
for key in list(h5_file.keys()):
    print('shape of {} is {}'.format(key, h5_file[key].shape))

select = np.sort(np.random.random_integers(0, 10000, 5))
pulse = h5_file['Time_pulse'][select, :, :]
waveform = pulse[:, 0:1, :] * np.exp(-1J * pulse[:, 1:2, :])
pulse_phase_recon = pulse[:, 2:, :].copy()
pulse_phase_recon[:, 1:] -= pulse_phase_recon[:, :-1]
waveform_sum = pulse[:, 0:1, :] * np.exp(-1J * pulse_phase_recon)
pulse_waveform = np.concatenate((pulse, waveform.real, waveform.imag, waveform_sum.real, waveform_sum.imag), axis=1)

cut = np.arange(start=0, stop=1000, step=10)
pulse_cut = h5_file['Time_pulse'][select, :, :][:, :, cut]
waveform_cut = pulse_cut[:, 0:1, :] * np.exp(-1J * pulse_cut[:, 1:2, :])
pulse_phase_recon_cut = pulse_cut[:, 2, :].copy()
pulse_phase_recon_cut[:, 1:] -= pulse_phase_recon_cut[:, :-1]
waveform_sum_cut = pulse_cut[:, 0:1, :] * np.exp(-1J * pulse_phase_recon_cut)
pulse_waveform_cut = np.concatenate(
    (pulse_cut, waveform_cut.real, waveform_cut.imag, waveform_sum_cut.real, waveform_sum_cut.imag), axis=1)

h5_file.close()

fig, ax = plt.subplots(nrows=int(pulse_waveform.shape[0]), ncols=int(pulse_waveform.shape[1]), figsize=(22, 17),
                       sharex=True)
grid = np.indices(dimensions=(int(pulse_waveform.shape[0]), int(pulse_waveform.shape[1])))
row = grid[0].flatten()
row_imag = row * 2
col = grid[1].flatten()
index = np.arange(pulse.shape[0])

for ro_i, ro, co in zip(row_imag, row, col):
    ax[ro, co].plot(np.arange(1000), pulse_waveform[ro, co, :], 'b', np.arange(start=0, stop=1000, step=10),
                    pulse_waveform_cut[ro, co, :], 'r')
    #ax[ro_i + 1, co].plot(np.arange(start=0, stop=1000, step=10), pulse_waveform_cut[ro, co, :])
'''
for ro_i, ro, co in zip(row_imag, row, col):
    ax[ro_i, co].plot(np.arange(1000), pulse_waveform[ro, co, :])
    ax[ro_i + 1, co].plot(np.arange(start=0, stop=1000, step=10), pulse_waveform_cut[ro, co, :])
'''
# fig.savefig('Images/KernelDensityEstimate.png', dpi=700)
