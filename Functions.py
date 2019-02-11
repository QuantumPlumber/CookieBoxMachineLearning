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


def input_hdf5_functor(transfer='reformed_spectra_final.hdf5', select=1000, batch_size=1):
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
    Spectra16_select = Spectra16[0:1000, ...]

    VN_coeff_select = VN_coeff[0:1000, ...]
    # VN_coeff_select_expand = np.concatenate((VN_coeff_select.real, VN_coeff_select.imag), axis=1)

    dataset = tf.data.Dataset.from_tensor_slices((Spectra16_select, VN_coeff_select))
    dataset = dataset.shuffle(Spectra16_select.shape[0]).repeat().batch(batch_size)

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
    ############### Define the Graph Structure

    ### It looks like feature columns are not only uncescessary but breaking the CNN
    ##net = tf.feature_column.input_layer(features, params['feature_columns'])

    # net = tf.reshape(features["sequences"], [-1, 32, 5])

    feat_shape = features.shape

    cookie_list = []
    for cookie in range(params['NUM_COOKIES']):
        net = features[:, cookie, ...]
        # for filters, window in params['CNN']:
        #    net = tf.layers.conv1d(inputs=net, filters=filters, kernel_size=window, activation=tf.nn.relu)
        # net = tf.layers.max_pooling1d(inputs=net, strides=1, pool_size=params['POOL'])
        for nodes in params['COOKIE_DENSE']:
            net = tf.layers.dense(inputs=net, units=nodes, activation=tf.nn.relu)
        cookie_list.append(net)

    cookie_box = tf.concat(values=cookie_list, axis=1)
    print(cookie_box.shape)
    for nodes in params['DENSE']:
        net = tf.layers.dense(inputs=cookie_box, units=nodes, activation=tf.nn.relu)

    net = tf.layers.dense(inputs=net, units=params['OUT'], activation=tf.nn.relu)
    print(net.shape)

    net = tf.layers.flatten(net)
    output = tf.cast(tf.complex(net[:, 0:100], net[:, 100:200]), dtype=tf.complex128)

    ################# Compute loss and metrics
    loss = tf.losses.absolute_difference(labels=labels, predictions=output)

    accuracy = tf.metrics.mean_absolute_error(labels=labels,
                                              predictions=output,
                                              name='acc_op')

    ############### Prediction mode.
    # predicted_classes = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {

            'output': output,

        }

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    ######## Evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    ######## Train mode
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
