import tensorflow as tf
import os
import numpy as np

mag_scale_factor = 100
phase_scale_factor = 1200 * np.pi

direct = './multilayer_cnn_3'
filename = direct + '/' + 'saved_model.h5'
try:
    keras_model
except NameError:
    print('no keras model in memory')
    if os.path.isfile(filename):
        print('re-loading model from: {}'.format(filename))
        keras_model = tf.keras.models.load_model(filepath=filename)
    else:
        print('could not find file at location: {}'.format(filename))
else:
    print('keras_model already instantiated')
