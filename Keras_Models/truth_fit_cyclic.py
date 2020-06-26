import tensorflow as tf
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

mag_scale_factor = 1
phase_scale_factor = 1


def data_generator(transfer='TF_train_wave_unwrapped_eggs.hdf5', batch_size=64, cut_bot=.8, cut_top=1., reps=1):
    '''
        Reformats transformed simulation data into TFRecord data format

        :param filename:
        :return:
        '''

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

    random_shuffled_index = np.arange(0, Spectra.shape[0])[
                            int(Spectra.shape[0] * cut_bot):int(Spectra.shape[0] * cut_top)]
    #np.random.shuffle(random_shuffled_index)
    random_shuffled_index = np.tile(random_shuffled_index, reps=reps)

    print((int(Spectra.shape[0] * cut_bot), int(Spectra.shape[0] * cut_top)))

    for batch in np.arange(start=0, stop=random_shuffled_index.shape[0], step=batch_size):
        sort_index = np.sort(np.unique(random_shuffled_index[batch: batch + batch_size]))

        magnitude = mag_scale_factor * Time_pulse[sort_index, :]
        phase = Phase_pulse[sort_index, :]
        complex_waveform = magnitude * np.exp(1j * phase)
        real_waveform = np.real(complex_waveform)
        imaginary_waveform = np.imag(complex_waveform)

        spect = Spectra[sort_index, ...]
        spect_pb = (spect[:, :, :, 0] ** 2 + spect[:, :, :, 1] ** 2)*1e9

        yield (spect_pb,
               {'real': real_waveform,
                'imaginary': imaginary_waveform})


transfer_filename = '../../AttoStreakSimulations/Data/TF_another_set.hdf5'

train_data = data_generator(transfer=transfer_filename,
                            batch_size=1,
                            cut_bot=.0,
                            cut_top=1.0,
                            reps=200)

test_data = data_generator(transfer=transfer_filename,
                           batch_size=1,
                           cut_bot=.0,
                           cut_top=1.0,
                           reps=200)

direct = './Cyclic_CNN_Truth'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=direct,
                                                      histogram_freq=100,
                                                      batch_size=1,
                                                      write_graph=True,
                                                      write_grads=False,
                                                      write_images=False)
'''
keras.callbacks.ModelCheckpoint(filepath,
                                monitor='val_loss',
                                verbose=0,
                                save_best_only=False,
                                save_weights_only=False,
                                mode='auto',
                                period=1)
'''

filename = direct + '/' + 'saved_model.h5'
try:
    keras_model
except NameError:
    print('no keras model in memory')
    if os.path.isfile(filename):
        keras_model = tf.keras.models.load_model(filepath=filename)
else:
    print('keras_model already instantiated')

history = keras_model.fit_generator(train_data,
                                    steps_per_epoch=int(1e3),
                                    epochs=10,
                                    verbose=1,
                                    #callbacks=[tensorboard_callback],
                                    validation_data=test_data,
                                    validation_steps=int(1e1),
                                    class_weight=None,
                                    max_queue_size=10,
                                    workers=1,
                                    use_multiprocessing=False)

tf.keras.models.save_model(model=keras_model, filepath=filename, overwrite=True)
# del keras_model

fig, axs = plt.subplots(6, 1, constrained_layout=True, figsize=(6, 18))
for i, key in enumerate(history.history):
    axs[i].plot(history.history[key])
    axs[i].set_title(key)

fig.savefig(direct + '/' + '/history.png', dpi=700)
