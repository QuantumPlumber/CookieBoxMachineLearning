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

print(Spectra.shape)

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
direct = 'multilayer_cnn_truth'
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

cut_bot = 0.8
cut_top = 1.0
num_spectra = 30

spectra_index = np.arange(0, Spectra.shape[0])[int(Spectra.shape[0] * cut_bot):int(Spectra.shape[0] * cut_top)]

num_spike = np.random.randint(low=1, high=5, size=num_spectra)

magnitude_list = []
phase_list = []
spect_pb_list = []
for num in num_spike:
    sort_index = np.sort(np.random.choice(spectra_index, num, replace=False))
    magnitude_list.append(mag_scale_factor * np.sum(Time_pulse[sort_index, :], axis=0, keepdims=True))
    phase_list.append(np.sum(Nonlinearphase_pulse[sort_index, :], axis=0, keepdims=True))
    spect = np.sum(Spectra[sort_index, ...], axis=0, keepdims=True)
    spect_pb_list.append((spect[:, :, :, 0] ** 2 + spect[:, :, :, 1] ** 2) * 1e9)

spectra = np.concatenate(spect_pb_list, axis=0)
mag_truth = np.concatenate(magnitude_list, axis=0)
phase_truth = np.concatenate(phase_list, axis=0)

h5_reformed.close()

predictions = keras_model.predict(spectra, batch_size=num_spectra, verbose=0)

fig, ax = plt.subplots(nrows=int(mag_truth.shape[0] / 3), ncols=int(2 * 3),
                       figsize=(22, int(mag_truth.shape[0] / 3) * 3))
grid = np.indices(dimensions=(int(mag_truth.shape[0] / 3), 3))
row = grid[0].flatten()
col = grid[1].flatten() * 2
index = np.arange(mag_truth.shape[0])
for ind, ro, co, mag_pred, phase_pred in zip(index, row, col, predictions[0], predictions[1]):
    # for ind, ro, co, phase_pred in zip(index, row, col, predictions):
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

fig.savefig('sampleWaveforms2.png', dpi=700)

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
