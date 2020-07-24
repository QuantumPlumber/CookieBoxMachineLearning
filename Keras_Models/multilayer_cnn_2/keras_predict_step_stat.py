import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py
import importlib
import os

mag_scale_factor = 100
phase_scale_factor = 30 * np.pi

# load data

#transfer = '../Data/unwrapped/Eggs/TF_train_wave_unwrapped_eggs.hdf5'
#transfer = '../Data/25Hit_unwrapped/Scrambled/TF_train_25Hits_eggs.hdf5'
#transfer = '../Data/unwrapped_step/TF_train_waveform_unwrapped_step_eggs.hdf5'
#transfer = '../Data/25Hit_unwrapped/Scrambled/TF_train_25Hits_step_eggs.hdf5'
#transfer = '../Data/25_hit_5-14-19/25_hit_5-14-19_convert.hdf5'
#transfer = '../Data/25_hit_5-14-19/large_kernel.hdf5'
transfer = '../Data/25_hit_5-14-19/TF_100hit_0-6pulse_convert.hdf5'

h5_reformed = h5py.File(transfer, 'r')
print(h5_reformed.keys())
for key in list(h5_reformed.keys()):
    print('shape of {} is {}'.format(key, h5_reformed[key].shape))

hits = h5_reformed['Hits'][...]

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
direct = 'multilayer_cnn_2'
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

cut_bot = int(Pulse_truth.shape[0] * .8)
cut_top = int(Pulse_truth.shape[0] * 1.)
num_spectra = 1000
num_spectra_plot = 30

cut = np.unique(np.random.random_integers(low=cut_bot, high=cut_top, size=num_spectra))
print(cut)
spectra = Spectra16[cut, ...]
ground_truther = Pulse_truth[cut, ...]
mag_truth = ground_truther[:, 0, :]
phase_truth = ground_truther[:, 1, :]
mag_truth = mag_truth * phase_scale_factor

# ground_truther[:, 1, 1:] -= ground_truther[:, 1, :-1]
# phase_truth = ground_truther*mag_truth

h5_reformed.close()

predictions = keras_model.predict(spectra, batch_size=num_spectra, verbose=0)
mag_pred = predictions[0]
phase_pred = predictions[1]

index = np.arange(ground_truther.shape[0])
abs_error_list = []
for ind in index:
    abs_error_list.append(np.sum(np.abs(mag_pred[ind] - mag_truth[ind]) + np.abs(phase_pred[ind] - phase_truth[ind])) / np.sum(
        mag_truth[ind] + phase_truth[ind]))
    #print('mse err0r = {}, abs error = {}'.format(mse_error, abs_diff))

sorted_indices = np.argsort(np.array(abs_error_list))
plot_index = sorted_indices[-(np.arange(num_spectra_plot)*3+1)].tolist()

fig, ax = plt.subplots(nrows=int(num_spectra_plot / 3), ncols=int(2 * 3),
                       figsize=(22, int(num_spectra_plot / 3) * 3))
grid = np.indices(dimensions=(int(num_spectra_plot / 3), 3))
row = grid[0].flatten()
col = grid[1].flatten() * 2
num_hits = 100
plot_dummy = np.arange(num_spectra_plot)
for dum, ind, ro, co in zip(plot_dummy, plot_index, row, col):
    ax[ro, co].plot(mag_truth[ind], 'b', mag_pred[ind], 'r')
    ax[ro, co + 1].plot(phase_truth[ind], 'b', phase_pred[ind], 'r')
    mse_error = np.sum(((mag_pred[ind] - mag_truth[ind]) ** 2 + (phase_pred[ind] - phase_truth[ind]) ** 2)) / 200.
    abs_diff = np.sum(np.abs(mag_pred[ind] - mag_truth[ind]) + np.abs(phase_pred[ind] - phase_truth[ind])) / np.sum(
        mag_truth[ind] + phase_truth[ind])
    print('mse err0r = {}, abs error = {}'.format(mse_error, abs_diff))
    # display(fig)
    np.savetxt('{}hit_mag_truth{}.txt'.format(num_hits, dum), mag_truth[ind])
    np.savetxt('{}hit_phase_truth{}.txt'.format(num_hits, dum), phase_truth[ind])
    np.savetxt('{}hit_mag_pred{}.txt'.format(num_hits, dum), mag_pred[ind])
    np.savetxt('{}hit_phase_pred{}.txt'.format(num_hits, dum), phase_pred[ind])

    # fig.savefig('Images/sampleWaveforms4.png', dpi= 700)
