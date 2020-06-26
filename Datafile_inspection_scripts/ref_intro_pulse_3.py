import h5py
import numpy as np
import matplotlib.pyplot as plt

filename = '../AttoStreakSimulations/TF_train_waveform.hdf5'
h5_file = h5py.File(filename, 'r')
print(h5_file.keys())
for key in list(h5_file.keys()):
    print('shape of {} is {}'.format(key, h5_file[key].shape))

select = np.sort(np.random.random_integers(0, 10000, 5))
cut = np.arange(start=0, stop=1000, step=10)
pulse_cut = h5_file['Time_pulse'][select, :, :][:, :, cut]
pulse = h5_file['Time_pulse'][select, :, :]
waveform = pulse[:, 0:1, :] * np.exp(-1J * pulse[:, 1:2, :])

# reconstruct pulse from accumulated phase
pulse_phase_recon = pulse[:, 2:, :].copy()
pulse_phase_recon[:, :, 1:] -= pulse_phase_recon[:, :, :-1]
pulse_phase_recon_div = pulse_phase_recon / pulse[:, 0:1, :]
waveform_sum = pulse[:, 0:1, :] * np.exp(-1J * pulse_phase_recon / pulse[:, 0:1, :])

# reconstruct pulse from downsampled waveforms using interpolation
fibonacci = np.tile(np.cumsum(np.arange(start=0, stop=10, step=1)), reps=100)
pulse_phase_cut = pulse_cut[:, 2:, :].copy()
pulse_phase_cut_recon = pulse_cut[:, 2:, :].copy()
pulse_phase_cut_recon[:, :, 1:] -= pulse_phase_cut_recon[:, :, :-1]
pulse_phase_cut_recon_repeat = np.repeat(pulse_phase_cut_recon / 55., repeats=10, axis=2)
pulse_phase_cut_recon_repeat_fib = pulse_phase_cut_recon_repeat * fibonacci
pulse_phase_cut_fib_add_recon = np.repeat(pulse_phase_cut, repeats=10, axis=2) + pulse_phase_cut_recon_repeat_fib
pulse_phase_cut_fib_add_recon[:, :, 1:] -= pulse_phase_cut_fib_add_recon[:, :, :-1]
'''
pulse_phase_cut_recon_interp = np.zeros(shape=(pulse_cut.shape[0], 1, 1000))
for i, phase in enumerate(pulse_phase_cut_recon[:, 0, :]):
    pulse_phase_cut_recon_interp[i, :, :] = np.interp(np.arange(start=0, stop=1000, step=1).astype('int32'),
                                                      np.arange(start=0, stop=1000, step=10).astype('int32'), phase)
'''
pulse_mag_cut_interp = np.zeros(shape=(pulse_cut.shape[0], 1, 1000))
for i, mag in enumerate(pulse_cut[:, 0, :]):
    pulse_mag_cut_interp[i, :, :] = np.interp(np.arange(start=0, stop=1000, step=1).astype('int32'),
                                              np.arange(start=0, stop=1000, step=10).astype('int32'), mag)
'''
pulse_phase_cut_interp_div = pulse_phase_cut_recon_interp / pulse_mag_cut_interp
waveform_sum_interp = pulse_mag_cut_interp * np.exp(-1J * pulse_phase_cut_interp_div)
'''
pulse_phase_cut_interp_div = pulse_phase_cut_fib_add_recon / pulse_mag_cut_interp
waveform_sum_interp = pulse_mag_cut_interp * np.exp(-1J * pulse_phase_cut_interp_div)

#pulse_waveform = np.concatenate((pulse_phase_recon, pulse_phase_cut_fib_add_recon), axis=1)

pulse_waveform = np.concatenate((pulse_phase_recon_div, pulse_phase_cut_interp_div), axis=1)

# pulse_waveform = np.concatenate(
#    (pulse, pulse_phase_cut_interp, pulse_phase_recon, pulse_phase_cut_interp_recon, pulse_phase_recon_div, pulse_phase_cut_interp_div),
#    axis=1)

# pulse_waveform = np.concatenate((waveform.real, waveform_sum_interp.real),
#                                  axis=1)
#pulse_waveform = np.concatenate((waveform.real, waveform.imag, waveform_sum_interp.real, waveform_sum_interp.imag),
#                                axis=1)
# pulse_waveform = np.concatenate((waveform.real, waveform.imag, waveform_sum.real, waveform_sum.imag), axis=1)
# pulse_waveform = np.concatenate((pulse, pulse_phase_recon, pulse_phase_recon_div), axis=1)


h5_file.close()

fig, ax = plt.subplots(nrows=int(pulse_waveform.shape[0]), ncols=int(pulse_waveform.shape[1]), figsize=(22, 17),
                       sharex=True)
grid = np.indices(dimensions=(int(pulse_waveform.shape[0]), int(pulse_waveform.shape[1])))
row = grid[0].flatten()
row_imag = row * 2
col = grid[1].flatten()
index = np.arange(pulse.shape[0])

for ro_i, ro, co in zip(row_imag, row, col):
    ax[ro, co].plot(np.arange(1000), pulse_waveform[ro, co, :], 'b')

    # ax[ro_i + 1, co].plot(np.arange(start=0, stop=1000, step=10), pulse_waveform_cut[ro, co, :])
'''
for ro_i, ro, co in zip(row_imag, row, col):
    ax[ro_i, co].plot(np.arange(1000), pulse_waveform[ro, co, :])
    ax[ro_i + 1, co].plot(np.arange(start=0, stop=1000, step=10), pulse_waveform_cut[ro, co, :])
'''
# fig.savefig('Images/KernelDensityEstimate.png', dpi=700)
