import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py
import importlib
import os

mag_scale_factor = 1
phase_scale_factor = 1

# load data

# transfer = '../../AttoStreakSimulations/Data/TF_workout_truth.hdf5'
# transfer = '../../AttoStreakSimulations/Data/TF_DokasHouse_truth.hdf5'
# transfer = '../../AttoStreakSimulations/Data/TF_SinglePulse_truth.hdf5'
# transfer = '../../AttoStreakSimulations/Data/TF_another_set.hdf5'
transfer = 'D:/CookieBox/AttoStreakSimulations/Data/TF_another_set.hdf5'

h5_reformed = h5py.File(transfer, 'r')

if 'Spectra' not in h5_reformed:
    raise Exception('No "Spectra" in file.')
else:
    Spectra = h5_reformed['Spectra']

if 'Nonlinearphase_pulse' not in h5_reformed:
    raise Exception('No "Phase_pulse" in file.')
else:
    Nonlinearphase_pulse = h5_reformed['Nonlinearphase_pulse']

if 'Time_pulse' not in h5_reformed:
    raise Exception('No "Time_pulse" in file.')
else:
    Time_pulse = h5_reformed['Time_pulse']

for key in list(h5_reformed.keys()):
    print('shape of {} is {}'.format(key, h5_reformed[key].shape))

# load model
#direct = 'multilayer_fc_truth'
direct = '.'
filename = direct + '/' + 'saved_model.h5'
try:
    keras_model
except NameError:
    print('no keras model in memory')
    if os.path.isfile(filename):
        print('loading model from file: {}'.format(filename))
        keras_model = tf.keras.models.load_model(filepath=filename)
    else:
        print('cannot find: {}'.format(filename))
else:
    print('keras_model already instantiated')

cut_bot = int(Time_pulse.shape[0] * 0.8)
cut_top = int(Time_pulse.shape[0] * 1.0)
num_spectra = 30

cut = np.unique(np.random.random_integers(low=cut_bot, high=cut_top, size=num_spectra))
print(cut)
spectra_true = Spectra[cut, ...]
spectra = spectra_true[:, :, :, 0] ** 2 + spectra_true[:, :, :, 1] ** 2
true_sum = np.sum(spectra)
spectra = spectra*1e9
mag_truth = Time_pulse[cut, ...] * mag_scale_factor
phase_truth = Nonlinearphase_pulse[cut, ...]


# ground_truther[:, 1, 1:] -= ground_truther[:, 1, :-1]
# phase_truth = ground_truther*mag_truth

h5_reformed.close()

predictions = keras_model.predict(spectra, batch_size=num_spectra, verbose=0)

fig, ax = plt.subplots(nrows=int(mag_truth.shape[0] / 3), ncols=int(2 * 3),
                       figsize=(22, int(mag_truth.shape[0] / 3) * 3))
grid = np.indices(dimensions=(int(mag_truth.shape[0] / 3), 3))
row = grid[0].flatten()
col = grid[1].flatten() * 2
index = np.arange(mag_truth.shape[0])
num_hits = 400
for ind, ro, co, mag_pred, phase_pred in zip(index, row, col, predictions[0], predictions[1]):
#for ind, ro, co, phase_pred in zip(index, row, col, predictions):
    ax[ro, co].plot(mag_truth[ind], 'b', mag_pred, 'r')
    ax[ro, co + 1].plot(phase_truth[ind], 'b', phase_pred, 'r')
    mse_error = np.sum(((mag_pred - mag_truth[ind]) ** 2 + (phase_pred - phase_truth[ind]) ** 2)) / 200.
    abs_diff = np.sum(np.abs(mag_pred - mag_truth[ind]) + np.abs(phase_pred - phase_truth[ind])) / np.sum(
        mag_truth[ind] + phase_truth[ind])
    print('mse err0r = {}, abs error = {}'.format(mse_error, abs_diff))
    # display(fig)
    # np.savetxt('{}hit_mag_truth{}.txt'.format(num_hits, ind), mag_truth[ind])
    # np.savetxt('{}hit_phase_truth{}.txt'.format(num_hits, ind), phase_truth[ind])
    # np.savetxt('{}hit_mag_pred{}.txt'.format(num_hits, ind), mag_pred)
    # np.savetxt('{}hit_phase_pred{}.txt'.format(num_hits, ind), phase_pred)

    # fig.savefig('Images/sampleWaveforms4.png', dpi= 700)

'''
fig, ax = plt.subplots(nrows=int(mag_truth.shape[0] / 3), ncols=int(2 * 3),
                       figsize=(22, int(mag_truth.shape[0] / 3) * 3))
grid = np.indices(dimensions=(int(mag_truth.shape[0] / 3), 3))
row = grid[0].flatten()
col = grid[1].flatten() * 2
index = np.arange(mag_truth.shape[0])
num_hits = 400
for ro, co, spect in zip(row, col, spectra_true):
    for spec in spect:
        ax[ro, co].plot((spec[:, 0]**2 + spec[:, 1]**2)/true_sum, 'b')
        ax[ro, co + 1].plot(spec[:, 1], 'r')
'''