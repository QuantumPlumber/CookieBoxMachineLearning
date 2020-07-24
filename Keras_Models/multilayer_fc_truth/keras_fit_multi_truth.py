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

    if 'Nonlinearphase_pulse' not in h5_reformed:
        raise Exception('No "Phase_pulse" in file.')
    else:
        Nonlinearphase_pulse = h5_reformed['Nonlinearphase_pulse']

    if 'Time_pulse' not in h5_reformed:
        raise Exception('No "Time_pulse" in file.')
    else:
        Time_pulse = h5_reformed['Time_pulse']

    spectra_index = np.arange(0, Spectra.shape[0])[int(Spectra.shape[0] * cut_bot):int(Spectra.shape[0] * cut_top)]
    seed_size = 1000
    seed_index = np.arange(0, seed_size)

    print((int(Spectra.shape[0] * cut_bot), int(Spectra.shape[0] * cut_top)))

    count = 0
    while True:

        if count % 1000 == 0:
            seed_pick = np.sort(np.random.choice(spectra_index, seed_size, replace=False))
            Time_pulse_seed = Time_pulse[seed_pick, :]
            Nonlinearphase_pulse_seed = Nonlinearphase_pulse[seed_pick, :]
            Spectra_seed = Spectra[seed_pick, ...]

        num_spike = np.random.randint(low=1, high=5, size=batch_size)

        magnitude_list = []
        phase_list = []
        spect_pb_list = []
        for num in num_spike:
            sort_index = np.sort(np.random.choice(seed_index, num, replace=False))
            magnitude_list.append(mag_scale_factor * np.sum(Time_pulse_seed[sort_index, :], axis=0, keepdims=True))
            phase_list.append(np.sum(Nonlinearphase_pulse_seed[sort_index, :], axis=0, keepdims=True))
            spect = np.sum(Spectra_seed[sort_index, ...], axis=0, keepdims=True)
            spect_pb_list.append((spect[:, :, :, 0] ** 2 + spect[:, :, :, 1] ** 2) * 1e9)

    count += 1
    yield (np.concatenate(spect_pb_list, axis=0),
           {'magnitude': np.concatenate(magnitude_list, axis=0), 'phase': np.concatenate(phase_list, axis=0)})


# transfer_filename = '../../AttoStreakSimulations/Data/TF_workout_truth.hdf5'
# transfer_filename = '../../AttoStreakSimulations/Data/TF_DokasHouse_truth.hdf5'
transfer_filename = '../../AttoStreakSimulations/Data/TF_another_set.hdf5'

train_data = data_generator(transfer=transfer_filename,
                            batch_size=64,
                            cut_bot=.0,
                            cut_top=.80,
                            reps=200)

transfer_filename = '../../AttoStreakSimulations/Data/TF_another_set.hdf5'

test_data = data_generator(transfer=transfer_filename,
                           batch_size=64,
                           cut_bot=.80,
                           cut_top=1.0,
                           reps=200)

# direct = './multilayer_cnn_truth'

direct = './multilayer_fc_truth'
'''
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=direct,
                                                      histogram_freq=1000,
                                                      batch_size=1,
                                                      write_graph=True,
                                                      write_grads=False,
                                                      write_images=False)
'''
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
                                    epochs=3,
                                    verbose=1,
                                    # callbacks=[tensorboard_callback],
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
