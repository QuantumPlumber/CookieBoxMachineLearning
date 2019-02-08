import tensorflow as tf
import numpy as np
import h5py
import numpy as np
from Functions import input_functor
from Functions import CNNmodel

data = np.random.rand(100, 16, 1920)
labels = np.random.rand(100, 100)
test_data = np.random.rand(100, 16, 1920)
test_labels = np.random.rand(100, 100)


'''
transfer = 'reformed.hdf5'
h5_reformed = h5py.File(transfer)

if 'Spectra16' not in h5_reformed:
    raise Exception('No "Spectra16" in file.')
else:
    Spectra16 = h5_reformed['Spectra16']
    print('Spectra16 shape: {}'.format(Spectra16.shape))

if 'VN_coeff' not in h5_reformed:
    raise Exception('No "VN_coeff" in file.')
else:
    VN_coeff = h5_reformed['VN_coeff']
    print('VN_coeff shape: {}'.format(VN_coeff.shape))

if 'Hits' not in h5_reformed:
    raise Exception('No "Hits" in file.')
else:
    Hits = h5_reformed['Hits']

# Get 1k random samples from the dataset.
index = np.arange(list(Spectra16.shape)[0])
np.random.shuffle(index)
index_cut = index[0:1000]
index_cut.sort()

print('gathering data from hdf5 file...')
data = Spectra16[index_cut, ...]
print('data size is {} bytes'.format(data.nbytes))
labels = VN_coeff[index_cut, ...]
print('labels size is {} bytes'.format(labels.nbytes))
print('data retrieved.')

h5_reformed.close()
'''

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
    input_fn=lambda: input_hdf5_functor(data=feattestdatanorm, labels=feattestlabelsnorm, batch_size=1),
    steps=10)
