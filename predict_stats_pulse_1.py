import numpy as np
import Functions as fn
import matplotlib.pyplot as plt
import h5py
import importlib

importlib.reload(fn)

h5_file = h5py.File('TF_train_wave_unwrapped.hdf5', 'r')

if 'Pulse_truth' not in h5_file:
    raise Exception('No "Pulse_truth" in file.')
else:
    Pulse_truth = h5_file['Pulse_truth']

for key in list(h5_file.keys()):
    print('shape of {} is {}'.format(key, h5_file[key].shape))

cuts = [[0, 6000], [6000, 12000], [12000, 18000], [18000, 24000], [24000, 30000]]
cuts = [[0, 6000], [6000, 12000]]
cuts = [[0, 6000], [6000, 12000]]


def cuts_generator(min_val=0, max_val=1000, step=100):
    for ii in np.arange(min_val, max_val-1, step):
        yield ([ii, ii + step])


min_val = 0
max_val = 100000
cuts = cuts_generator(min_val=min_val, max_val=max_val, step=5000)

tot_error = np.zeros(shape=Pulse_truth.shape)

for run, cut in enumerate(cuts):
    predictions = classifier.predict(
        input_fn=lambda: fn.predict_hdf5_functor(transfer=transfer, select=cut,
                                                 batch_size=1))

    ground_truther = Pulse_truth[cut[0]:cut[1], :, :]
    mag_truth = ground_truther[:, 0, :]
    mag_predict = np.zeros_like(mag_truth)
    phase_truth = ground_truther[:, 1, :] * ground_truther[:, 0, :]
    phase_predict = np.zeros_like(phase_truth)

    predict_truther = np.zeros_like(ground_truther, dtype='complex64')

    i = 0
    for predict in predictions:
        mag_predict[i, :] = predict['output'][0:100]
        phase_predict[i, :] = predict['output'][100:200]
        i = i + 1

    tot_error[cut[0]:cut[1], 0, :] = ((mag_truth - mag_predict) ** 2) / 200
    tot_error[cut[0]:cut[1], 1, :] = ((phase_truth - phase_predict) ** 2) / 200

    print('completed run {}'.format(run))

h5_file.close()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))
im = ax[0].pcolormesh(tot_error[min_val:max_val-1, 0, :])
fig.colorbar(im, ax=ax[0])
im = ax[1].pcolormesh(tot_error[min_val:max_val-1, 1, :])
fig.colorbar(im, ax=ax[1])

# fig.savefig('Images/percent_errors2.png', dpi= 700)
