import numpy as np
import Functions as fn
import matplotlib.pyplot as plt
import h5py
import importlib

importlib.reload(fn)

# transfer = 'reformed_spectra_densesapce_safe.hdf5'
# transfer = 'reformed_TF_train_mp_1.hdf5'
# transfer = 'reformed_TF_train_mp_1_quarter.hdf5'
transfer = 'reformed_TF_train_widegate.hdf5'

h5_reformed = h5py.File(transfer, 'r')
if 'VN_coeff' not in h5_reformed:
    raise Exception('No "VN_coeff" in file.')
else:
    VN_coeff = h5_reformed['VN_coeff']

for key in list(h5_reformed.keys()):
    print('shape of {} is {}'.format(key, h5_reformed[key].shape))

cut_bot = 00
cut_top = 10

ground_truther = VN_coeff[cut_bot:cut_top, ...]
mag_truth = np.abs(ground_truther)
phase_truth = np.angle(ground_truther)


h5_reformed.close()

predictions = classifier.predict(
    input_fn=lambda: fn.predict_hdf5_functor(transfer=transfer,
                                             select=(cut_bot, cut_top),
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

    # display(fig)

    # fig.savefig('Images/sampleWaveforms4.png', dpi= 700)
