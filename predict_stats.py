import numpy as np
import Functions as fn
import matplotlib.pyplot as plt
import h5py
import importlib

importlib.reload(fn)

transfer = 'reformed_spectra_densesapce_safe.hdf5'
h5_reformed = h5py.File(transfer, 'r')
if 'VN_coeff' not in h5_reformed:
    raise Exception('No "VN_coeff" in file.')
else:
    VN_coeff = h5_reformed['VN_coeff']

for key in list(h5_reformed.keys()):
    print('shape of {} is {}'.format(key, h5_reformed[key].shape))

cuts = [[0, 6000], [6000, 12000], [12000, 18000], [18000, 24000], [24000, 30000]]

mag_error = np.zeros(shape=VN_coeff.shape)
phase_error = np.zeros(shape=VN_coeff.shape)

for run, cut in enumerate(cuts):
    predictions = classifier.predict(
        input_fn=lambda: fn.predict_hdf5_functor(transfer='reformed_spectra_densesapce_safe.hdf5', select=cut,
                                                 batch_size=1))

    ground_truther = VN_coeff[cut, ...]
    predict_truther = np.zeros_like(ground_truther, dtype='complex64')
    i = 0
    for predict in predictions:
        predict_truther[i, ...] = predict[0:100] + 1J * predict[100:200]

    abs_ground_truther = np.abs(ground_truther)
    abs_predict_truther = np.abs(predict_truther)
    theta_ground_truther = np.angle(ground_truther)
    theta_predict_truther = np.angle(predict_truther)
    mag_error[cut, ...] = np.abs(abs_ground_truther - abs_predict_truther) / abs_ground_truther
    phase_error = np.abs(theta_predict_truther - theta_ground_truther) / theta_ground_truther

    print('completed run {}'.format(run))

h5_reformed.close()

'''
for ground, predict in zip(ground_truther, predictions):
    fig = plt.figure(figsize=(8, 8))
    display(plt.plot(ground.real, 'b', predict['output'][0:100], 'r'))
    fig = plt.figure(figsize=(8, 8))
    display(plt.plot(ground.imag, 'b', predict['output'][100:200], 'r'))
'''
