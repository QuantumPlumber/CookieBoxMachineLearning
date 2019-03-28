import numpy as np
import Functions as fn
import matplotlib.pyplot as plt
import h5py
import importlib

importlib.reload(fn)

# transfer = 'reformed_spectra_densesapce_safe.hdf5'
transfer = 'reformed_TF_train_mp_1.hdf5'
h5_reformed = h5py.File(transfer, 'r')
if 'VN_coeff' not in h5_reformed:
    raise Exception('No "VN_coeff" in file.')
else:
    VN_coeff = h5_reformed['VN_coeff']

for key in list(h5_reformed.keys()):
    print('shape of {} is {}'.format(key, h5_reformed[key].shape))

cuts = [[0, 6000], [6000, 12000], [12000, 18000], [18000, 24000], [24000, 30000]]
cuts = [[0, 6000], [6000, 12000]]
cuts = [[0, 6000], [6000, 12000]]
cuts = [[0,5000]]

mag_error = np.zeros(shape=VN_coeff.shape)
phase_error = np.zeros(shape=VN_coeff.shape)

for run, cut in enumerate(cuts):
    predictions = classifier.predict(
        input_fn=lambda: fn.predict_hdf5_functor(transfer=transfer, select=cut,
                                                 batch_size=1))

    ground_truther = VN_coeff[cut[0]:cut[1], ...]
    mag_truth = np.abs(ground_truther)
    mag_predict = np.zeros_like(mag_truth)
    phase_truth = np.angle(ground_truther)
    phase_predict = np.zeros_like(phase_truth)

    predict_truther = np.zeros_like(ground_truther, dtype='complex64')

    i = 0
    for predict in predictions:
        mag_predict[i, ...] = predict['output'][0:100]
        phase_predict[i, ...] = predict['output'][100:200]
        i = i + 1

    mag_error[cut[0]:cut[1], ...] = np.abs(mag_truth - mag_predict)
    phase_error[cut[0]:cut[1], ...] = np.abs(phase_truth - phase_predict)

    print('completed run {}'.format(run))

h5_reformed.close()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))
im = ax[0].pcolormesh(mag_error[cut[0]:cut[1]])
fig.colorbar(im, ax=ax[0])
im = ax[1].pcolormesh(phase_error[cut[0]:cut[1]])
fig.colorbar(im, ax=ax[1])

# fig.savefig('Images/percent_errors2.png', dpi= 700)
