import numpy as np
import Functions as fn
import matplotlib.pyplot as plt
import h5py
import importlib

importlib.reload(fn)

transfer = 'TF_train_waveform_convert.hdf5'

h5_reformed = h5py.File(transfer, 'r')

if 'Pulse_truth' not in h5_reformed:
    raise Exception('No "Pulse_truth" in file.')
else:
    Pulse_truth = h5_reformed['Pulse_truth']

for key in list(h5_reformed.keys()):
    print('shape of {} is {}'.format(key, h5_reformed[key].shape))

cut_bot = 0000
cut_top = 100000

cut = np.unique(np.random.random_integers(low=cut_bot, high=cut_top, size=6))
print(cut)
ground_truther = Pulse_truth[cut, ...]
mag_truth = ground_truther[:, 0, :]
phase_truth = ground_truther[:, 1, :]

h5_reformed.close()

predictions = classifier.predict(
    input_fn=lambda: fn.predict_hdf5_functor(transfer=transfer,
                                             select=cut,
                                             batch_size=1))

fig, ax = plt.subplots(nrows=int(ground_truther.shape[0] / 3), ncols=int(2 * 3),
                       figsize=(22, int(ground_truther.shape[0] / 3) * 3))
grid = np.indices(dimensions=(int(ground_truther.shape[0] / 3), 3))
row = grid[0].flatten()
col = grid[1].flatten() * 2
index = np.arange(ground_truther.shape[0])
for ind, ro, co, predict in zip(index, row, col, predictions):
    mag_pred = predict['output'][0:100]
    phase_pred = predict['output'][100:200]
    ax[ro, co].plot(mag_truth[ind], 'b', mag_pred, 'r')
    ax[ro, co + 1].plot(phase_truth[ind], 'b', phase_pred, 'r')
    print(np.sum(((mag_pred - mag_truth[ind]) ** 2 + (phase_pred - phase_truth[ind]) ** 2)) / 200.)
    # display(fig)

    # fig.savefig('Images/sampleWaveforms4.png', dpi= 700)
