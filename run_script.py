
import tensorflow as tf
import numpy as np

from Functions import input_functor
from Functions import CNNmodel

testdatanorm = np.random.rand(100, 32, 5)
testlabelsnorm = np.random.rand(100, 1)
traindatanorm = np.random.rand(100, 32, 5)
trainlabelsnorm = np.random.rand(100, 1)

feattestdatanorm = np.random.rand(100, 32, 5)
feattestlabelsnorm = np.random.rand(100, 1)
feattraindatanorm = np.random.rand(100, 32, 5)
feattrainlabelsnorm = np.random.rand(100, 1)

classifier = tf.estimator.Estimator(

    model_fn=CNNmodel,

    params={

        # 'feature_columns': the_feature_column,

        # Layers.

        'CNN': [[32, 5], [32, 5]],  # Convolutional layers
        'POOL': 24,  # Global Pooling Label
        'DENSE': [32, 16, 8],  # Dense layers
        'OUT': 1  # output dimensions

    })

classifier.train(
    input_fn=lambda: input_functor(datanorm=feattestdatanorm, labelsnorm=feattestlabelsnorm, batch_size=32),
    steps=10)
