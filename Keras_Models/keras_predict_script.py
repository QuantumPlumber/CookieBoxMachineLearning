import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py
import importlib
import os

# load data

# transfer = 'TF_train_wave_unwrapped.hdf5'

transfer = '../Data/unwrapped/Eggs/TF_train_wave_unwrapped_eggs.hdf5'
h5_reformed = h5py.File(transfer, 'r')

if 'Spectra16' not in h5_reformed:
    raise Exception('No "Spectra16" in file.')
else:
    Spectra16 = h5_reformed['Spectra16']

if 'Pulse_truth' not in h5_reformed:
    raise Exception('No "Pulse_truth" in file.')
else:
    Pulse_truth = h5_reformed['Pulse_truth']

for key in list(h5_reformed.keys()):
    print('shape of {} is {}'.format(key, h5_reformed[key].shape))

# load model
direct = './multilayer_cnn_big'
filename = direct + '/' + 'saved_model.h5'
mag_scale_factor = 100
phase_scale_factor = 1200 * np.pi
try:
    keras_model
except NameError:
    print('no keras model in memory')
    if os.path.isfile(filename):
        keras_model = tf.keras.models.load_model(filepath=filename)
else:
    print('keras_model already instantiated')

cut_bot = Pulse_truth.shape[0] * .8
cut_top = Pulse_truth.shape[0] * 1.
num_spectra = 10

cut = np.unique(np.random.random_integers(low=cut_bot, high=cut_top, size=num_spectra))
print(cut)
spectra = Spectra16[cut, ...]
ground_truther = Pulse_truth[cut, ...]
mag_truth = ground_truther[:, 0, :]
phase_truth = ground_truther[:, 1, :] * mag_truth
mag_truth = mag_truth*phase_scale_factor

# ground_truther[:, 1, 1:] -= ground_truther[:, 1, :-1]
# phase_truth = ground_truther*mag_truth

h5_reformed.close()

predictions = keras_model.predict(spectra, batch_size=num_spectra, verbose=0)

fig, ax = plt.subplots(nrows=int(ground_truther.shape[0] / 3), ncols=int(2 * 3),
                       figsize=(22, int(ground_truther.shape[0] / 3) * 3))
grid = np.indices(dimensions=(int(ground_truther.shape[0] / 3), 3))
row = grid[0].flatten()
col = grid[1].flatten() * 2
index = np.arange(ground_truther.shape[0])
for ind, ro, co, mag_pred, phase_pred in zip(index, row, col, predictions[0], predictions[1]):
    ax[ro, co].plot(mag_truth[ind], 'b', mag_pred, 'r')
    ax[ro, co + 1].plot(phase_truth[ind], 'b', phase_pred, 'r')
    mse_error = np.sum(((mag_pred - mag_truth[ind]) ** 2 + (phase_pred - phase_truth[ind]) ** 2)) / 200.
    abs_diff = np.sum(np.abs(mag_pred - mag_truth[ind]) + np.abs(phase_pred - phase_truth[ind])) / np.sum(
        mag_truth[ind] + phase_truth[ind])
    print('mse err0r = {}, abs error = {}'.format(mse_error, abs_diff))
    # display(fig)

    # fig.savefig('Images/sampleWaveforms4.png', dpi= 700)
