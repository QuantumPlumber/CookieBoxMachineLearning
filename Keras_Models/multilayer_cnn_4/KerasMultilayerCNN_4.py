import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import plot_model

spectra16 = tf.keras.layers.Input(shape=(16, 100), name='16-eToF-Spectra')
spectra16_reshaped = tf.keras.layers.Reshape(target_shape=(16, 100, 1))(spectra16)  # reshape for CNN


def slicer(inputs, i):
    # return tf.slice(inputs[0], begin=(0, inputs[1], 0, 0), size=(-1, 0, -1, -1))
    return (inputs[:, i, :, :])


slicing = tf.keras.layers.Lambda(slicer)
conv1 = tf.keras.layers.Conv1D(filters=10, kernel_size=5, padding='valid')  # no zero padding
pool1 = tf.keras.layers.MaxPooling1D(pool_size=5, strides=2)
conv2 = tf.keras.layers.Conv1D(filters=10, kernel_size=5, padding='valid')  # no zero padding
pool2 = tf.keras.layers.MaxPooling1D(pool_size=5, strides=2)
conv3 = tf.keras.layers.Conv1D(filters=10, kernel_size=5, padding='valid')  # no zero padding
pool3 = tf.keras.layers.MaxPooling1D(pool_size=5, strides=1)
spectra16_conv_list = []

for spect in range(16):
    slicing.arguments = {'i': spect}
    sp = slicing(spectra16_reshaped)
    con = conv1(sp)
    pool = pool1(con)
    con = conv2(pool)
    pool = pool2(con)
    con = conv3(pool)
    pool = pool3(con)
    spectra16_conv_list.append(pool)

net = tf.keras.layers.Concatenate()(spectra16_conv_list)
net = tf.keras.layers.Flatten()(net)

dense_network = [1760, 500, 500]
for nodes in dense_network[:-1]:
    net = tf.keras.layers.Dense(units=nodes)(net)

output = [100, 100]
mag_scale_factor = 100
phase_scale_factor = 1200 * np.pi

mag = tf.keras.layers.Lambda(lambda x: x * phase_scale_factor, name='magnitude')(
    tf.keras.layers.Dense(units=output[0])(net))
phase = tf.keras.layers.Lambda(lambda x: x * phase_scale_factor, name='phase')(
    tf.keras.layers.Dense(units=output[1])(net))

keras_model = tf.keras.Model(inputs=spectra16, outputs=[mag, phase])

adadelta = tf.keras.optimizers.Adadelta(lr=.01, rho=0.95, epsilon=1e-8, decay=0.0)
keras_model.compile(optimizer=adadelta,
                    loss={'magnitude': 'mean_squared_error', 'phase': 'mean_squared_error'},
                    loss_weights=[1., 1.])


direct = './multilayer_cnn_4'
filename = direct+'/'+'model.png'
plot_model(keras_model, to_file=filename, show_shapes=True)

filename = direct+'/'+'model.json'
f = open(filename,'w')
f.write(keras_model.to_json())
f.close()
