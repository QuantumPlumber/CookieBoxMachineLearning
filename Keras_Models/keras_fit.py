import tensorflow as tf
import numpy as np
import h5py


def data_generator(transfer='TF_train_wave_unwrapped_eggs.hdf5', batch_size=64, cut_bot=.8, cut_top=1.):
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
    np.random.shuffle(random_shuffled_index)

    print((int(Spectra16.shape[0] * cut_bot),int(Spectra16.shape[0] * cut_top)))

    for batch in np.arange(start=0, stop=random_shuffled_index.shape[0], step=batch_size):
        yield (Spectra16[np.sort(random_shuffled_index[batch: batch + batch_size]), ...],
               {'magnitude': Pulse_truth[np.sort(random_shuffled_index[batch: batch + batch_size]), 0, :],
                'magphase': Pulse_truth[np.sort(random_shuffled_index[batch: batch + batch_size]), 1, :]})



train_data = data_generator(transfer='../TF_train_wave_unwrapped_eggs.hdf5', batch_size=64, cut_bot=.0, cut_top=.8)
test_data = data_generator(transfer='../TF_train_wave_unwrapped_eggs.hdf5', batch_size=64, cut_bot=.8, cut_top=1.0)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./tensorboard',
                                                      histogram_freq=1,
                                                      batch_size=64,
                                                      write_graph=True,
                                                      write_grads=False,
                                                      write_images=False)

keras_model.fit_generator(train_data,
                          steps_per_epoch=int(1e3),
                          epochs=1,
                          verbose=1,
                          callbacks=[tensorboard_callback],
                          validation_data=test_data,
                          validation_steps=int(1e1),
                          class_weight=None,
                          max_queue_size=10,
                          workers=1,
                          use_multiprocessing=False,
                          shuffle=True)
'''
keras_model.fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0,
                validation_data=None,
                shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None,
                validation_steps=None,
                validation_freq=1)
'''