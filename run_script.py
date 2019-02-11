import tensorflow as tf
import numpy as np
import Functions as fn

testdatanorm = np.random.rand(100,16,100)
testlabelsnorm = np.random.rand(100,100)

classifier = tf.estimator.Estimator(

    model_fn=fn.CNNmodel,
    model_dir='Model',
    params={

        # 'feature_columns': the_feature_column,

        # Layers.
        'NUM_COOKIES': 16,
        'COOKIE_DENSE': [100, 50, 10],
        'CNN': [[32, 10], [32, 10]],  # Convolutional layers
        'POOL': 80,  # Global Pooling Label
        'DENSE': [160, 200, 200],  # Dense layers
        'OUT': 200  # output dimensions

    })

classifier.train(
    #input_fn=lambda: fn.input_functor(datanorm=testdatanorm, labelsnorm=testlabelsnorm, batch_size=1),
    input_fn=lambda: fn.input_hdf5_functor(transfer='reformed_spectra_final.hdf5', batch_size=1),
    steps=1)
