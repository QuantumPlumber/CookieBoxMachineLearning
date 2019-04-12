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

cut = np.unique(np.random.random_integers(low=cut_bot, high=cut_top, size=36))
print(cut)
ground_truther = Pulse_truth[cut, ...]
mag_truth = ground_truther[:, 0, :]
mag_truth_interp = np.zeros(shape=(mag_truth.shape[0], 1000))
for i, mag in enumerate(mag_truth):
    mag_truth_interp[i, :] = np.interp(np.arange(start=0, stop=1000, step=1), np.arange(start=0, stop=1000, step=10),
                                       mag)

phase_truth = ground_truther[:, 1, :]
phase_truth_interp = np.zeros(shape=(phase_truth.shape[0], 1000))
for i, phase in enumerate(phase_truth):
    phase_truth_interp[i, :] = np.interp(np.arange(start=0, stop=1000, step=1), np.arange(start=0, stop=1000, step=10),
                                         phase)

phase_truth_interp[:, 1:] -= phase_truth_interp[:, :-1]
truth_truth = mag_truth_interp * np.exp(-1J * phase_truth_interp / mag_truth_interp)

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
    mag_pred = np.interp(np.arange(start=0, stop=1000, step=1), np.arange(start=0, stop=1000, step=10),
                         predict['output'][0:100])
    phase_pred = np.interp(np.arange(start=0, stop=1000, step=1), np.arange(start=0, stop=1000, step=10),
                           predict['output'][100:200])
    phase_pred[1:] -= phase_pred[:-1]
    pred_pred = mag_pred * np.exp(-1J * phase_pred / mag_pred)
    # ax[ro, co].plot(truth_truth.real[ind], 'b')
    # ax[ro, co + 1].plot(truth_truth.imag[ind], 'b')
    ax[ro, co].plot(truth_truth.real[ind], 'b', pred_pred.real, 'r')
    ax[ro, co + 1].plot(truth_truth.imag[ind], 'b', pred_pred.imag, 'r')
    print(np.sum(((mag_pred - mag_truth_interp[ind]) ** 2 + (phase_pred - phase_truth_interp[ind]) ** 2)) / 200.)
    # display(fig)

    # fig.savefig('Images/sampleWaveforms4.png', dpi= 700)
