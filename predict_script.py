import numpy as np
import Functions as fn
import matplotlib.pyplot as plt
import h5py

transfer='reformed_spectra_final.hdf5'
h5_reformed = h5py.File(transfer, 'r')
if 'VN_coeff' not in h5_reformed:
    raise Exception('No "VN_coeff" in file.')
else:
    VN_coeff = h5_reformed['VN_coeff']
ground_truth = VN_coeff[3000:3002, ...]
h5_reformed.close()

predictions = classifier.predict(
    input_fn=lambda: fn.predict_hdf5_functor(transfer='reformed_spectra_final.hdf5', batch_size=1))

for ground, predict in zip(ground_truth,predictions):
    fig = plt.figure(figsize=(8, 8))
    display(plt.plot(ground_truth[0].real,'b', predict['output'].real,'r'))
    fig = plt.figure(figsize=(8, 8))
    display(plt.plot(ground_truth[0].imag,'b', predict['output'].imag,'r'))