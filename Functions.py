import tensorflow as tf
# from tensorflow import keras
# tf.enable_eager_execution()
# print(tf.executing_eagerly)

import numpy as np
import matplotlib.pyplot as plt

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


def CNNmodel(features, labels, mode, params):
    ############### Define the Graph Structure

    ### It looks like feature columns are not only uncescessary but breaking the CNN
    ##net = tf.feature_column.input_layer(features, params['feature_columns'])

    # net = tf.reshape(features["sequences"], [-1, 32, 5])

    net = features;

    for filters, window in params['CNN']:
        net = tf.layers.conv1d(inputs=net, filters=filters, kernel_size=window, activation=tf.nn.relu)

    net = tf.layers.max_pooling1d(inputs=net, strides=1, pool_size=params['POOL'])

    for nodes in params['DENSE']:
        net = tf.layers.dense(inputs=net, units=nodes, activation=tf.nn.relu)

    output = tf.layers.dense(inputs=net, units=params['OUT'], activation=tf.nn.relu)[:, 0]

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
