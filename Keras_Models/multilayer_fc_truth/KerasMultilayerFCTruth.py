import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import plot_model

spectra16 = tf.keras.layers.Input(shape=(16, 1000), name='16-eToF-Spectra')
#spectra16_reshaped = tf.keras.layers.Reshape(target_shape=(16, 1000, 1))(spectra16)  # reshape for CNN


def slicer(inputs, i):
    # return tf.slice(inputs[0], begin=(0, inputs[1], 0, 0), size=(-1, 0, -1, -1))
    return (inputs[:, i, :])


activation_string = None

slicing = tf.keras.layers.Lambda(slicer)
dense1 = tf.keras.layers.Dense(units=1000, activation='relu')
dense2 = tf.keras.layers.Dense(units=333, activation='relu')
dense3 = tf.keras.layers.Dense(units=111, activation='relu')

spectra16_conv_list = []
for spect in range(16):
    slicing.arguments = {'i': spect}
    sp = slicing(spectra16)
    den = dense1(sp)
    den = dense2(den)
    den = dense3(den)
    spectra16_conv_list.append(den)

net = tf.keras.layers.Concatenate()(spectra16_conv_list)
net = tf.keras.layers.Flatten()(net)

redux = 1760
net_linear = tf.keras.layers.Dense(units=redux, activation=None)(net)
net_nonlinear = tf.keras.layers.Dense(units=redux, activation='tanh')(net)
net = tf.keras.layers.Multiply()([net_linear, net_nonlinear])

dense_network = [333, 333, 333]
for nodes in dense_network:
    net_linear = tf.keras.layers.Dense(units=nodes, activation=None)(net)
    net_nonlinear = tf.keras.layers.Dense(units=nodes, activation='tanh')(net)
    net = tf.keras.layers.Multiply()([net_linear, net_nonlinear])

output = [1000, 1000]
mag_scale_factor = 800
phase_scale_factor = 1

mag = tf.keras.layers.Lambda(lambda x: x * phase_scale_factor, name='magnitude')(
    tf.keras.layers.Dense(units=output[0], activation=None)(net))
phase = tf.keras.layers.Lambda(lambda x: x * phase_scale_factor, name='phase')(
    tf.keras.layers.Dense(units=output[1], activation=None)(net))

keras_model = tf.keras.Model(inputs=spectra16, outputs=[mag, phase])

RMSprop = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
# Adadelta = tf.keras.optimizers.Adadelta(lr=.01, rho=0.95, epsilon=1e-8, decay=0.0)

keras_model.compile(optimizer=RMSprop,
                    loss={'magnitude': 'mean_squared_error', 'phase': 'mean_squared_error'},
                    loss_weights=[1., 1.])




# direct = './multilayer_cnn_2'
# direct = './multilayer_fc_truth'
direct = '.'
filename = direct + '/' + 'model.png'
plot_model(keras_model, to_file=filename, show_shapes=True)

filename = direct + '/' + 'model.json'
f = open(filename, 'w')
f.write(keras_model.to_json())
f.close()
