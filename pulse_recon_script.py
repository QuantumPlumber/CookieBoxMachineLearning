import numpy as np
import Functions as fn
import simclass
import matplotlib.pyplot as plt
import h5py
import importlib

importlib.reload(fn)
importlib.reload(simclass)

runningMan = simclass.Simulacrum(filename='TF_training_densespace.hdf5')
runningMan.computeSetup(I_p=200)
runningMan.time_pulse(pulse_center=40.3889, atto_pulse_width=-.500, t_0=1.8, chirp=-5, N_step=int(1e3),
                      dt=runningMan.dt)
runningMan.VN_projector(basis=runningMan.time_waveform, filename=runningMan.filename)

transfer = 'reformed_spectra_densesapce_safe.hdf5'
h5_reformed = h5py.File(transfer, 'r')
if 'VN_coeff' not in h5_reformed:
    raise Exception('No "VN_coeff" in file.')
else:
    VN_coeff = h5_reformed['VN_coeff']

for key in list(h5_reformed.keys()):
    print('shape of {} is {}'.format(key, h5_reformed[key].shape))

cut_bot = 7500
cut_top = 7518

ground_truther = VN_coeff[cut_bot:cut_top, ...]
h5_reformed.close()

predictions = classifier.predict(
    input_fn=lambda: fn.predict_hdf5_functor(transfer='reformed_spectra_densesapce_safe.hdf5',
                                             select=(cut_bot, cut_top),
                                             batch_size=1))

fig, ax = plt.subplots(nrows=int(ground_truther.shape[0] / 3), ncols=3, figsize=(22, 17))
grid = np.indices(dimensions=(int(ground_truther.shape[0] / 3), 3))
row = grid[0].flatten()
col = grid[1].flatten() * 2
index = np.arange(ground_truther.shape[0])
for ind, ro, co, predict in zip(index, row, col, predictions):
    ground_pulse = runningMan.VN_reconstructor(coeff=np.reshape(ground_truther[ind], newshape=(10, 10)),
                                               filename='TF_training_densespace.hdf5')

    VN_predict = predict['output'][0:100] + 1J * predict['output'][100:200]
    predict_pulse = runningMan.VN_reconstructor(coeff=np.reshape(VN_predict, newshape=(10, 10)),
                                                filename='TF_training_densespace.hdf5')

    ax[ro, co].plot(ground_truther[ind].real, 'b', predict['output'][0:100], 'r')

display(fig)