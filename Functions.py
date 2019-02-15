import tensorflow as tf
# from tensorflow import keras
# tf.enable_eager_execution()
# print(tf.executing_eagerly)

import numpy as np
import h5py

# import pandas as pd
# from scipy.optimize import curve_fit

print(tf.__version__)


def input_functor(datanorm, labelsnorm, batch_size):
    long = int(datanorm.shape[0])
    # long = int(datanorm.shape[0])
    dataset = tf.data.Dataset.from_tensor_slices((datanorm, labelsnorm))
    dataset = dataset.shuffle(long).repeat().batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()
    return dataset


def input_hdf5_functor(transfer='reformed_spectra_final.hdf5', select=(0, 1000), batch_size=1):
    h5_reformed = h5py.File(transfer, 'r')

    if 'Spectra16' not in h5_reformed:
        raise Exception('No "Spectra16" in file.')
    else:
        Spectra16 = h5_reformed['Spectra16']

    if 'VN_coeff' not in h5_reformed:
        raise Exception('No "VN_coeff" in file.')
    else:
        VN_coeff = h5_reformed['VN_coeff']

    random_sample = np.random.random_integers(0, Spectra16.shape[0] - 1, select)
    random_sorted = np.sort(random_sample)
    Spectra16_select = Spectra16[select[0]:select[1], ...]

    VN_coeff_select = VN_coeff[select[0]:select[1], ...]
    VN_coeff_select_expand = np.concatenate((VN_coeff_select.real, VN_coeff_select.imag), axis=1)

    dataset = tf.data.Dataset.from_tensor_slices((Spectra16_select, VN_coeff_select_expand))
    dataset = dataset.shuffle(Spectra16_select.shape[0]).repeat().batch(batch_size)

    h5_reformed.close()
    return dataset.make_one_shot_iterator().get_next()
    # return dataset


def evaluate_hdf5_functor(transfer='reformed_spectra_final.hdf5', select=(0, 1000), batch_size=1000):
    h5_reformed = h5py.File(transfer, 'r')

    if 'Spectra16' not in h5_reformed:
        raise Exception('No "Spectra16" in file.')
    else:
        Spectra16 = h5_reformed['Spectra16']

    if 'VN_coeff' not in h5_reformed:
        raise Exception('No "VN_coeff" in file.')
    else:
        VN_coeff = h5_reformed['VN_coeff']

    random_sample = np.random.random_integers(0, Spectra16.shape[0] - 1, select)
    random_sorted = np.sort(random_sample)
    Spectra16_select = Spectra16[select[0]:select[1], ...]

    VN_coeff_select = VN_coeff[select[0]:select[1], ...]
    VN_coeff_select_expand = np.concatenate((VN_coeff_select.real, VN_coeff_select.imag), axis=1)

    dataset = tf.data.Dataset.from_tensor_slices((Spectra16_select, VN_coeff_select_expand))
    dataset = dataset.shuffle(Spectra16_select.shape[0]).batch(batch_size)

    h5_reformed.close()
    return dataset.make_one_shot_iterator().get_next()
    # return dataset


def predict_hdf5_functor(transfer='reformed_spectra_final.hdf5', select=(3000, 3001), batch_size=1):
    h5_reformed = h5py.File(transfer, 'r')

    if 'Spectra16' not in h5_reformed:
        raise Exception('No "Spectra16" in file.')
    else:
        Spectra16 = h5_reformed['Spectra16']

    if 'VN_coeff' not in h5_reformed:
        raise Exception('No "VN_coeff" in file.')
    else:
        VN_coeff = h5_reformed['VN_coeff']

    #Spectra16_select = Spectra16[select[0]:select[1], ...]
    #VN_coeff_select = VN_coeff[select[0]:select[1], ...]

    Spectra16_select = Spectra16[select, ...]


    # VN_coeff_select_expand = np.concatenate((VN_coeff_select.real, VN_coeff_select.imag), axis=1)

    dataset = tf.data.Dataset.from_tensor_slices((Spectra16_select))
    dataset = dataset.batch(batch_size)

    h5_reformed.close()
    return dataset.make_one_shot_iterator().get_next()
    # return dataset


def input_hdf5_functor_map(data, labels, batch_size):
    data_shape = data.shape
    labels_shape = labels.shape
    data.sort()  # limit 300 entries max per cookie.
    labels = np.reshape(labels, (labels_shape[0], labels_shape[1] * labels_shape[2]))

    dataset = tf.data.Dataset.from_tensor_slices((data[:, :, -300:-1], labels))
    dataset = dataset.shuffle(data_shape[0]).repeat().batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()
    return dataset


def map_function_hdf5():
    energies = tf.linspace(start=0, stop=100, num=100)
    #  this is done so much easier in numpy..


def CNNmodel(features, labels, mode, params):
    cookie_list = []
    for cookie in range(params['NUM_COOKIES']):
        net = features[:, cookie, ...]
        # for filters, window in params['CNN']:
        #    net = tf.layers.conv1d(inputs=net, filters=filters, kernel_size=window, activation=tf.nn.relu)
        # net = tf.layers.max_pooling1d(inputs=net, strides=1, pool_size=params['POOL'])
        for nodes in params['COOKIE_DENSE']:
            net = tf.layers.dense(inputs=net, units=nodes, activation=tf.nn.tanh)
        cookie_list.append(net)

    cookie_box = tf.concat(values=cookie_list, axis=1)
    for nodes in params['DENSE']:
        net = tf.layers.dense(inputs=cookie_box, units=nodes, activation=tf.nn.tanh)

    net = tf.layers.dense(inputs=net, units=params['OUT'], activation=tf.nn.tanh)

    net = tf.layers.flatten(net)
    norm = tf.reduce_max(tf.sqrt(net[:, 0:100] ** 2 + net[:, 100:200] ** 2), axis=1, keepdims=True)
    print(norm)
    net = net / norm  # normalize explicitly
    print(net)
    output = net
    # output = tf.cast(tf.complex(net[:, 0:100], net[:, 100:200]), dtype=tf.complex128)

    ############### Prediction mode.
    # predicted_classes = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {

            'output': output,

        }

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.mean_squared_error(labels=tf.cast(labels, dtype=tf.float32), predictions=output)
    accuracy = tf.metrics.mean_squared_error(labels=tf.cast(labels, dtype=tf.float32), predictions=output)

    ######## Evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops={'accuracy': accuracy})

    ######## Train mode

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
