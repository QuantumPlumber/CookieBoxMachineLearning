import numpy as np
import Functions as fn
import matplotlib.pyplot as plt
import h5py
import importlib

importlib.reload(fn)

transfer = 'reformed_spectra_safe.hdf5'
h5_reformed = h5py.File(transfer, 'r')
if 'VN_coeff' not in h5_reformed:
    raise Exception('No "VN_coeff" in file.')
else:
    VN_coeff = h5_reformed['VN_coeff']

for key in list(h5_reformed.keys()):
    print('shape of {} is {}'.format(key, h5_reformed[key].shape))

ground_truther = VN_coeff[29000:29002, ...]
ground_truther = ground_truther / np.max(np.abs(ground_truther))
h5_reformed.close()

predictions = classifier.predict(
    input_fn=lambda: fn.predict_hdf5_functor(transfer='reformed_spectra_safe.hdf5', select=(29000, 29002), batch_size=1))


for ground, predict in zip(ground_truther, predictions):
    fig = plt.figure(figsize=(8, 8))
    display(plt.plot(ground.real, 'b', predict['output'][0:100], 'r'))
    fig = plt.figure(figsize=(8, 8))
    display(plt.plot(ground.imag, 'b', predict['output'][100:200], 'r'))
