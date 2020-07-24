import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py
import importlib
import os

# load data

transfer = 'D:/CookieBox/AttoStreakSimulations/Data/TF_another_set.hdf5'

h5_reformed = h5py.File(transfer, 'r')

if 'Spectra' not in h5_reformed:
    raise Exception('No "Spectra" in file.')
else:
    Spectra = h5_reformed['Spectra']

if 'Phase_pulse' not in h5_reformed:
    raise Exception('No "Phase_pulse" in file.')
else:
    Phase_pulse = h5_reformed['Phase_pulse']

if 'Time_pulse' not in h5_reformed:
    raise Exception('No "Time_pulse" in file.')
else:
    Time_pulse = h5_reformed['Time_pulse']

for key in list(h5_reformed.keys()):
    print('shape of {} is {}'.format(key, h5_reformed[key].shape))

# load model
#direct = 'multilayer_cnn_truth'
filename = 'saved_model.h5'
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

cut_bot = int(Time_pulse.shape[0] * 0.)
cut_top = int(Time_pulse.shape[0] * .8)
num_spectra = 30

cut = np.unique(np.random.random_integers(low=cut_bot, high=cut_top, size=num_spectra))
print(cut)
spectra_true = Spectra[cut, ...]
spectra = spectra_true[:, :, :, 0] ** 2 + spectra_true[:, :, :, 1] ** 2
spectra = spectra / np.reshape(np.max(spectra, axis=(1, 2)), newshape=[num_spectra, 1, 1])
magnitude = Time_pulse[cut, ...]
mag_truth = magnitude

# ground_truther[:, 1, 1:] -= ground_truther[:, 1, :-1]
# phase_truth = ground_truther*mag_truth

h5_reformed.close()

predictions = keras_model.predict(spectra, batch_size=num_spectra, verbose=0)

fig, ax = plt.subplots(nrows=int(mag_truth.shape[0] / 3), ncols=int(3),
                       figsize=(22, int(mag_truth.shape[0] / 3) * 3))
grid = np.indices(dimensions=(int(mag_truth.shape[0] / 3), 3))
row = grid[0].flatten()
col = grid[1].flatten()
index = np.arange(mag_truth.shape[0])
num_hits = 400
for ind, ro, co, mag_pred in zip(index, row, col, predictions):
    ax[ro, co].plot(mag_truth[ind], 'b', mag_pred, 'r')

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