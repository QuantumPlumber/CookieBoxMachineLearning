import tensorflow as tf
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

mag_scale_factor = 100
phase_scale_factor = 1200 * np.pi


def data_generator(transfer='TF_train_wave_unwrapped_eggs.hdf5', batch_size=64, cut_bot=.8, cut_top=1., reps=20):
    '''
        Reformats transformed simulation data into TFRecord data format

        :param filename:
        :return:
        '''

    h5_reformed = h5py.File(transfer, 'r')

    if 'Spectra16' not in h5_reformed:
        raise Exception('No "Spectra16" in file.')
    else:
        Spectra16 = h5_reformed['Spectra16']

    if 'Pulse_truth' not in h5_reformed:
        raise Exception('No "Pulse_truth" in file.')
    else:
        Pulse_truth = h5_reformed['Pulse_truth']

    random_shuffled_index = np.arange(0, Spectra16.shape[0])[
                            int(Spectra16.shape[0] * cut_bot):int(Spectra16.shape[0] * cut_top)]
    random_shuffled_index = np.tile(random_shuffled_index, reps=reps)
    # np.random.shuffle(random_shuffled_index)

    print((int(random_shuffled_index.shape[0] * cut_bot), int(random_shuffled_index.shape[0] * cut_top)))

    for batch in np.arange(start=0, stop=random_shuffled_index.shape[0], step=batch_size):
        magnitude = phase_scale_factor * Pulse_truth[np.sort(random_shuffled_index[batch: batch + batch_size]), 0, :]
        phase = Pulse_truth[np.sort(random_shuffled_index[batch: batch + batch_size]), 1, :]
        magphase = magnitude / phase_scale_factor * phase
        yield (Spectra16[np.sort(random_shuffled_index[batch: batch + batch_size]), ...],
               {'magnitude': magnitude,
                'phase': magphase})


#transfer_filename = '../Data/unwrapped/Eggs/TF_train_wave_unwrapped_eggs.hdf5'

transfer_filename = '../Data/unwrapped_step/TF_train_waveform_unwrapped_step_eggs.hdf5'
train_data = data_generator(transfer=transfer_filename,
                            batch_size=64,
                            cut_bot=.0,
                            cut_top=.8)
test_data = data_generator(transfer=transfer_filename,
                           batch_size=64,
                           cut_bot=.8,
                           cut_top=1.0)

# direct = './multilayer_cnn'
# direct = './multilayer_cnn_big'
direct = './multilayer_cnn_2'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=direct,
                                                      histogram_freq=1,
                                                      batch_size=64,
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
                                    epochs=20,
                                    verbose=1,
                                    callbacks=[tensorboard_callback],
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
