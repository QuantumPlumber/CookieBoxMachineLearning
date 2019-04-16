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
mag = tf.keras.layers.Dense(units=output[0])(net)
phase_scale_factor = 600 * np.pi
phase = tf.keras.layers.Dense(units=output[1])(net)

keras_model = tf.keras.Model(inputs=spectra16, outputs=[mag, phase])

adadelta = tf.keras.optimizers.Adadelta(lr=.01, rho=0.95, epsilon=1e-8, decay=0.0)
keras_model.compile(optimizer=adadelta, loss='mean_squared_error', loss_weights=[phase_scale_factor, 1.])

plot_model(keras_model, to_file='model.png')

keras_model.fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None,
    shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None,
    validation_freq=1)
