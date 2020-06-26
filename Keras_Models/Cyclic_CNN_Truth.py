import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import plot_model

spectra16 = tf.keras.layers.Input(shape=(16, 1000), name='16-eToF-Spectra')
spectra16_reshaped = tf.keras.layers.Reshape(target_shape=(16, 1000, 1))(spectra16)  # reshape for CNN

'''
def slicer(inputs, i):
    # return tf.slice(inputs[0], begin=(0, inputs[1], 0, 0), size=(-1, 0, -1, -1))
    return (inputs[:, i, :, :])


activation_string = None

slicing = tf.keras.layers.Lambda(slicer)
# dense1 = tf.keras.layers.Dense(units=1000, activation='relu')
# dense2 = tf.keras.layers.Dense(units=100, activation='relu')


conv1 = tf.keras.layers.Conv1D(filters=30, kernel_size=10, padding='valid',
                               activation=activation_string)  # no zero padding
pool1 = tf.keras.layers.MaxPooling1D(pool_size=11, strides=5)
conv2 = tf.keras.layers.Conv1D(filters=30, kernel_size=5, padding='valid',
                               activation=activation_string)  # no zero padding
pool2 = tf.keras.layers.MaxPooling1D(pool_size=11, strides=5)
conv3 = tf.keras.layers.Conv1D(filters=30, kernel_size=10, padding='valid',
                               activation=activation_string)  # no zero padding
pool3 = tf.keras.layers.MaxPooling1D(pool_size=11, strides=5)



spectra16_conv_list = []
for spect in range(16):
    slicing.arguments = {'i': spect}
    sp = slicing(spectra16_reshaped)
    # den_linear = tf.keras.layers.Dense(units=1000, activation='linear')(sp)
    # den_nonlinear = tf.keras.layers.Dense(units=1000, activation=None)
    # den = tf.keras.layers.Dense(units=100, activation='linear')(den_linear)
    con = conv1(sp)
    pool = pool1(con)
    con = conv2(pool)
    pool = pool2(con)
    con = conv3(pool)
    pool = pool3(con)
    spectra16_conv_list.append(pool)
'''

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

'''
redux = 1760
net_linear = tf.keras.layers.Dense(units=redux, activation=None)(net)
net_nonlinear = tf.keras.layers.Dense(units=redux, activation='tanh')(net)
net = tf.keras.layers.Multiply()([net_linear, net_nonlinear])
'''

def pi_scale(input):
    return 2*np.pi*input/tf.reduce_sum(input, axis=1, keepdims=True)

dense_network = [300, 300]
for nodes in dense_network:
    net_linear = tf.keras.layers.Dense(units=nodes, activation=None)(net)
    net_scale = tf.keras.layers.Lambda(pi_scale)(net)
    net_nonlinear = tf.keras.layers.Dense(units=nodes, activation=tf.sin)(net_scale)
    net = tf.keras.layers.Multiply()([net_linear, net_nonlinear])

output = [1000, 1000]
real_wave = tf.keras.layers.Dense(units=output[0], activation=None, name='real')(net)
imag_wave = tf.keras.layers.Dense(units=output[1], activation=None, name='imaginary')(net)

keras_model = tf.keras.Model(inputs=spectra16, outputs=[real_wave, imag_wave])

#RMSprop = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
Adadelta = tf.keras.optimizers.Adadelta(lr=.01, rho=0.95, epsilon=1e-8, decay=0.0)

def cyclic_loss(y_true, y_pred):
   #return tf.reduce_mean(y_true*tf.acos(y_pred/(y_true+.001)))
   return tf.reduce_mean(y_true - y_pred)

keras_model.compile(optimizer=Adadelta,
                    loss={'real': cyclic_loss, 'imaginary': cyclic_loss},
                    loss_weights=[1., 1.])

# direct = './multilayer_cnn_2'
direct = './Cyclic_CNN_Truth'
filename = direct + '/' + 'model.png'
plot_model(keras_model, to_file=filename, show_shapes=True)

filename = direct + '/' + 'model.json'
f = open(filename, 'w')
f.write(keras_model.to_json())
f.close()
