import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import plot_model

spectra16 = tf.keras.layers.Input(shape=(16, 100), name='16-eToF-Spectra')
spectra16_reshaped = tf.keras.layers.Reshape(target_shape=(16, 100, 1))(spectra16)  # reshape for CNN


def slicer(inputs, i):
    # return tf.slice(inputs[0], begin=(0, inputs[1], 0, 0), size=(-1, 0, -1, -1))
    return (inputs[:, i, :, :])


slicing = tf.keras.layers.Lambda(slicer)
conv1 = tf.keras.layers.Conv1D(filters=12, kernel_size=10, padding='valid')  # no zero padding
pool1 = tf.keras.layers.MaxPooling1D(pool_size=10, strides=5)
spectra16_conv_list = []

for spect in range(16):
    slicing.arguments = {'i': spect}
    sp = slicing(spectra16_reshaped)
    con = conv1(sp)
    pool = pool1(con)
    spectra16_conv_list.append(pool)

net = tf.keras.layers.Concatenate()(spectra16_conv_list)
net = tf.keras.layers.Flatten()(net)

dense_network = [2304, 1152, 1152, 500, 500]
for nodes in dense_network[:-1]:
    net = tf.keras.layers.Dense(units=nodes)(net)

output = [100, 100]
mag = tf.keras.layers.Dense(units=output[0], name='magnitude')(net)
phase_scale_factor = 600 * np.pi
phase = phase_scale_factor * tf.keras.layers.Dense(units=output[1])(net)
phase_dot = tf.keras.layers.Multiply(name='magphase')([mag, phase])

keras_model = tf.keras.Model(inputs=spectra16, outputs=[mag, phase_dot])

adadelta = tf.keras.optimizers.Adadelta(lr=.01, rho=0.95, epsilon=1e-8, decay=0.0)
keras_model.compile(optimizer=adadelta,
                    loss={'magnitude': 'mean_squared_error', 'magphase': 'mean_squared_error'},
                    loss_weights=[phase_scale_factor, 1.])

plot_model(keras_model, to_file='model.png')
