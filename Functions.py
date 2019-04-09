import tensorflow as tf
# from tensorflow import keras
# tf.enable_eager_execution()
# print(tf.executing_eagerly)

import numpy as np
import h5py

# import pandas as pd
# from scipy.optimize import curve_fit

print(tf.__version__)


def TFR_map_func(sequence_example):
    '''
    This map function returns the data in (input,label) pairs. A new dataset is created

    :param sequence_example:
    :return:
    '''
    context_features = {'VN_coeff': tf.FixedLenFeature(shape=(200), dtype=tf.float32)}
    sequence_features = {'spectra': tf.FixedLenSequenceFeature(shape=(100), dtype=tf.float32)}
    data = tf.parse_single_sequence_example(
        sequence_example,
        context_features=context_features,
        sequence_features=sequence_features
    )
    return data[1]['spectra'], data[0]['VN_coeff']


def input_TFR_functor(TFRecords_file_list=[], long=100000, repeat=1, batch_size=64):
    filenames = tf.data.Dataset.from_tensor_slices(TFRecords_file_list)
    dataset = tf.data.TFRecordDataset(filenames)

    dataset = dataset.map(TFR_map_func)

    dataset = dataset.shuffle(long).repeat(count=repeat).batch(batch_size=batch_size)
    return dataset.make_one_shot_iterator().get_next()


def input_hdf5_functor(transfer='reformed_spectra_final.hdf5', select=(0, 1000), batch_size=64):
    h5_reformed = h5py.File(transfer, 'r')

    if 'Spectra16' not in h5_reformed:
        raise Exception('No "Spectra16" in file.')
    else:
        Spectra16 = h5_reformed['Spectra16']

    if 'VN_coeff' not in h5_reformed:
        raise Exception('No "VN_coeff" in file.')
    else:
        VN_coeff = h5_reformed['VN_coeff']

    # random_sample = np.random.random_integers(0, Spectra16.shape[0] - 1, select)
    # random_sorted = np.sort(random_sample)

    # selections = np.arange(select[0], select[1], 10)
    # Spectra16_select = Spectra16[selections, ...]
    # VN_coeff_select = VN_coeff[selections, ...]
    # VN_coeff_select_expand = np.concatenate((VN_coeff_select.real, VN_coeff_select.imag), axis=1)

    # Spectra16_select = Spectra16[select[0]:select[1], ...]
    # VN_coeff_select = VN_coeff[select[0]:select[1], ...]
    # VN_coeff_select_expand = np.concatenate((VN_coeff_select.real, VN_coeff_select.imag), axis=1)

    Spectra16_select = Spectra16[select, ...]
    VN_coeff_select = VN_coeff[select, ...]
    VN_coeff_select_expand = np.concatenate((np.abs(VN_coeff_select), np.angle(VN_coeff_select)), axis=1)

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

    # Spectra16_select = Spectra16[select[0]:select[1], ...]
    # VN_coeff_select = VN_coeff[select[0]:select[1], ...]

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
    # return dataset


def map_function_hdf5():
    energies = tf.linspace(start=0, stop=100, num=100)
    #  this is done so much easier in numpy..


def CNNmodel(features, labels, mode, params):
    cookie_list = []
    for cookie in range(params['NUM_COOKIES']):
        net = tf.reshape(features[:, cookie, ...], shape=(-1, 100, 1))
        if cookie == 0:
            for filters, window in params['CNN']:
                net = tf.layers.conv1d(inputs=net,
                                       filters=filters,
                                       kernel_size=window,
                                       activation=tf.nn.tanh,
                                       name='conv1D_{}'.format(filters),
                                       reuse=False)
            net = tf.layers.max_pooling1d(inputs=net,
                                          strides=params['POOL'][0],
                                          pool_size=params['POOL'][1]
                                          )

            cookie_list.append(tf.layers.flatten(net))
        else:
            for filters, window in params['CNN']:
                net = tf.layers.conv1d(inputs=net,
                                       filters=filters,
                                       kernel_size=window,
                                       activation=tf.nn.tanh,
                                       name='conv1D_{}'.format(filters),
                                       reuse=True)
            net = tf.layers.max_pooling1d(inputs=net,
                                          strides=params['POOL'][0],
                                          pool_size=params['POOL'][1]
                                          )

            cookie_list.append(tf.layers.flatten(net))

    cookie_box = tf.concat(values=cookie_list, axis=1)

    for nodes in params['DENSE']:
        net = tf.layers.dense(inputs=cookie_box, units=nodes, activation=tf.nn.tanh)

    net = tf.layers.dense(inputs=net, units=params['OUT'], activation=tf.nn.tanh)

    net = tf.layers.flatten(net)

    norm_mag = tf.reduce_max(tf.abs(net[:, 0:100]), axis=1, keepdims=True)
    #mag = net[:, 0:100] / norm_mag
    mag = net[:, 0:100]
    norm_phase = tf.reduce_max(tf.abs(net[:, 100:200]), axis=1, keepdims=True)
    phase = np.pi * tf.abs(net[:, 100:200])
    output = tf.concat((mag, phase), axis=1)

    # norm = tf.reduce_max(tf.sqrt(net[:, 0:100] ** 2 + net[:, 100:200] ** 2), axis=1, keepdims=True)
    # net = net / norm  # normalize explicitly
    # output = net

    ############### Prediction mode.
    # predicted_classes = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {

            'output': output,

        }

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    labels = tf.concat((tf.cast(labels, dtype=tf.float32)[:, 0:100],
                        tf.abs(tf.cast(labels, dtype=tf.float32)[:, 100:200])),
                       axis=1)
    loss = tf.losses.mean_squared_error(labels=labels, predictions=output)
    accuracy = tf.metrics.mean_squared_error(labels=tf.cast(labels, dtype=tf.float32), predictions=output)

    ######## Evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops={'accuracy': accuracy})

    ######## Train mode

    # optimizer = tf.train.AdagradOptimizer(learning_rate=.01)
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=.01)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def CNNmodelMagPhase(features, labels, mode, params):
    cookie_list = []
    for cookie in range(params['NUM_COOKIES']):
        net = features[:, cookie, ...]
        # for filters, window in params['CNN']:
        #    net = tf.layers.conv1d(inputs=net, filters=filters, kernel_size=window, activation=tf.nn.relu)
        # net = tf.layers.max_pooling1d(inputs=net, strides=1, pool_size=params['POOL'])
        for nodes in params['COOKIE_DENSE']:
            net = tf.layers.dense(inputs=net, units=nodes, activation=tf.nn.tanh)
            # kernel_initializer=tf.initializers.random_uniform
        cookie_list.append(net)

    cookie_box = tf.concat(values=cookie_list, axis=1)
    for nodes in params['DENSE']:
        net = tf.layers.dense(inputs=cookie_box, units=nodes, activation=tf.nn.tanh)

    net = tf.layers.dense(inputs=net, units=params['OUT'], activation=tf.nn.tanh)

    net = tf.layers.flatten(net)

    # normalize magnitude and phase
    norm_mag = tf.reduce_max(tf.abs(net[:, 0:100]), axis=1, keepdims=True)
    mag = net[:, 0:100] / norm_mag
    norm_phase = tf.reduce_max(tf.abs(net[:, 0:100]), axis=1, keepdims=True)
    phase = np.pi * net[:, 100:200]

    # print(net)
    output = tf.concat((mag, phase), axis=1)

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

    optimizer = tf.train.AdagradOptimizer(learning_rate=.01)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def FC_large_sym_model(features, labels, mode, params):
    # bias and kernel clipping constraints
    kernel_clip_top = 100.0051
    kernel_clip_bot = -100.0051
    bias_clip_top = .000001
    bias_clip_bot = -.000001

    cookie_list = []
    for cookie in range(params['NUM_COOKIES']):
        net = features[:, cookie, ...]
        if cookie == 0:
            for name, nodes in enumerate(params['COOKIE_DENSE']):
                net = tf.layers.dense(inputs=net, units=nodes, activation=tf.nn.tanh,
                                      name='layer_{}'.format(name),
                                      reuse=False,
                                      kernel_constraint=lambda x: tf.clip_by_value(x, clip_value_min=kernel_clip_bot,
                                                                                   clip_value_max=kernel_clip_top),
                                      bias_constraint=lambda x: tf.clip_by_value(x, clip_value_min=bias_clip_bot,
                                                                                 clip_value_max=bias_clip_top))
                # kernel_initializer=tf.initializers.random_uniform
            cookie_list.append(net)
        else:
            for name, nodes in enumerate(params['COOKIE_DENSE']):
                net = tf.layers.dense(inputs=net, units=nodes, activation=tf.nn.tanh,
                                      name='layer_{}'.format(name),
                                      reuse=True,
                                      kernel_constraint=lambda x: tf.clip_by_value(x, clip_value_min=kernel_clip_bot,
                                                                                   clip_value_max=kernel_clip_top),
                                      bias_constraint=lambda x: tf.clip_by_value(x, clip_value_min=bias_clip_bot,
                                                                                 clip_value_max=bias_clip_top))
                # kernel_initializer=tf.initializers.random_uniform
            cookie_list.append(net)

    cookie_box = tf.concat(values=cookie_list, axis=1)
    for nodes in params['DENSE']:
        net = tf.layers.dense(inputs=cookie_box,
                              units=nodes,
                              activation=tf.nn.tanh,
                              kernel_constraint=lambda x: tf.clip_by_value(x, clip_value_min=kernel_clip_bot,
                                                                           clip_value_max=kernel_clip_top),
                              bias_constraint=lambda x: tf.clip_by_value(x, clip_value_min=bias_clip_bot,
                                                                         clip_value_max=bias_clip_top))
    net = tf.layers.dense(inputs=net,
                          units=params['OUT'],
                          activation=tf.nn.tanh,
                          kernel_constraint=lambda x: tf.clip_by_value(x, clip_value_min=kernel_clip_bot,
                                                                       clip_value_max=kernel_clip_top),
                          bias_constraint=lambda x: tf.clip_by_value(x, clip_value_min=bias_clip_bot,
                                                                     clip_value_max=bias_clip_top))

    net = tf.layers.flatten(net)

    # normalize magnitude and phase
    norm_mag = tf.reduce_max(tf.abs(net[:, 0:100]), axis=1, keepdims=True)
    mag = net[:, 0:100] / norm_mag
    norm_phase = tf.reduce_max(tf.abs(net[:, 100:200]), axis=1, keepdims=True)
    phase = np.pi * (net[:, 100:200] / norm_phase - 1)

    # print(net)
    output = tf.concat((mag, phase), axis=1)
    # loss_vec = tf.concat((mag, phase*mag))
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
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'accuracy': accuracy})

    ######## Train mode

    optimizer = tf.train.AdagradOptimizer(learning_rate=.01)
    # optimizer = tf.train.AdamOptimizer(learning_rate=.1, epsilon=1.0)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def FC_sym_model_ri(features, labels, mode, params):
    cookie_list = []
    for cookie in range(params['NUM_COOKIES']):
        net = features[:, cookie, ...]
        if cookie == 0:
            for name, nodes in enumerate(params['COOKIE_DENSE']):
                net = tf.layers.dense(inputs=net, units=nodes, activation=tf.nn.tanh, name='layer_{}'.format(name),
                                      reuse=False)
                # kernel_initializer=tf.initializers.random_uniform
            cookie_list.append(net)
        else:
            for name, nodes in enumerate(params['COOKIE_DENSE']):
                net = tf.layers.dense(inputs=net, units=nodes, activation=tf.nn.tanh, name='layer_{}'.format(name),
                                      reuse=True)
                # kernel_initializer=tf.initializers.random_uniform
            cookie_list.append(net)

    cookie_box = tf.concat(values=cookie_list, axis=1)
    for nodes in params['DENSE']:
        net = tf.layers.dense(inputs=cookie_box, units=nodes, activation=tf.nn.tanh)

    net = tf.layers.dense(inputs=net, units=params['OUT'], activation=tf.nn.tanh)

    net = tf.layers.flatten(net)
    # norm = tf.reduce_max(tf.sqrt(net[:, 0:100] ** 2 + net[:, 100:200] ** 2), axis=1, keepdims=True)
    # net = net / norm  # normalize explicitly
    output = net

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

    optimizer = tf.train.AdagradOptimizer(learning_rate=.0001)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
