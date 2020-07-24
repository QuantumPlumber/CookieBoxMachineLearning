import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import plot_model

spectra16 = tf.keras.layers.Input(shape=(16, 1000), name='16-eToF-Spectra')
spectra16_reshaped = tf.keras.layers.Reshape(target_shape=(16, 1000, 1))(spectra16)  # reshape for CNN

def slicer(inputs, i):
    # return tf.slice(inputs[0], begin=(0, inputs[1], 0, 0), size=(-1, 0, -1, -1))
    return (inputs[:, i, :])

activation_string = None

slicing = tf.keras.layers.Lambda(slicer)
dense1 = tf.keras.layers.Dense(units=1000, activation='tanh')
dense2 = tf.keras.layers.Dense(units=333, activation='tanh')
dense3 = tf.keras.layers.Dense(units=111, activation='tanh')

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

def pi_scale(input):
    return 2*np.pi*input/tf.reduce_sum(input, axis=1, keepdims=True)

dense_network = [300, 300]
for nodes in dense_network:
    net_linear = tf.keras.layers.Dense(units=nodes, activation=None)(net)
    #net_scale = tf.keras.layers.Lambda(pi_scale)(net)
    net_nonlinear = tf.keras.layers.Dense(units=nodes, activation='tanh')(net)
    net = tf.keras.layers.Multiply()([net_linear, net_nonlinear])

output = [1000, 1000]
real_wave = tf.keras.layers.Dense(units=output[0], activation=None, name='real')(net)
#imag_wave = tf.keras.layers.Dense(units=output[1], activation=None, name='imaginary')(net)

#keras_model = tf.keras.Model(inputs=spectra16, outputs=[real_wave, imag_wave])
keras_model = tf.keras.Model(inputs=spectra16, outputs=[real_wave])

#RMSprop = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
Adadelta = tf.keras.optimizers.Adadelta(lr=.01, rho=0.95, epsilon=1e-8, decay=0.0)

def cyclic_loss(y_true, y_pred):
   #return tf.reduce_mean(y_true*tf.acos(y_pred/(y_true+.001)))
   return tf.reduce_mean(tf.abs(y_true - y_pred))

keras_model.compile(optimizer=Adadelta,
                    #loss={'real': cyclic_loss, 'imaginary': cyclic_loss},
                    loss={'real': cyclic_loss},
                    loss_weights=[1.])


filename = 'model.png'
plot_model(keras_model, to_file=filename, show_shapes=True)

filename = 'model.json'
f = open(filename, 'w')
f.write(keras_model.to_json())
f.close()
