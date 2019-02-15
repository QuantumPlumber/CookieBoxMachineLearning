import numpy as np
import sys

sys.path.append('../AttoStreakSimulations')

import Functions as fn
import simclass
import matplotlib.pyplot as plt
import h5py
import importlib

importlib.reload(fn)
importlib.reload(simclass)

VN_file = '../AttoStreakSimulations/close_time.hdf5'
'''
runningMan = simclass.Simulacrum(filename=VN_file)
runningMan.computeSetup(I_p=200)
runningMan.time_pulse(pulse_center=40.3889, atto_pulse_width=-.500, t_0=1.8, chirp=-5, N_step=int(1e3),
                      dt=runningMan.dt)
runningMan.VN_projector(basis=runningMan.time_waveform, filename=runningMan.filename)
'''
transfer = 'reformed_spectra_densesapce_safe.hdf5'
h5_reformed = h5py.File(transfer, 'r')
if 'VN_coeff' not in h5_reformed:
    raise Exception('No "VN_coeff" in file.')
else:
    VN_coeff = h5_reformed['VN_coeff']

for key in list(h5_reformed.keys()):
    print('shape of {} is {}'.format(key, h5_reformed[key].shape))

# cut_bot = 7500
# cut_top = 7518
# select = np.arange(start=600, stop=30000, step=3000)
choice = np.random.choice(np.arange(0, 30000), size=1000, replace=False)
choice.sort()

# ground_truther = VN_coeff[cut_bot:cut_top, ...]
ground_truther = VN_coeff[choice, ...]

predictions = classifier.predict(
    input_fn=lambda: fn.predict_hdf5_functor(transfer='reformed_spectra_densesapce_safe.hdf5',
                                             select=choice,
                                             batch_size=1))

losses = []
for ground, predict in zip(ground_truther, predictions):
    loss = np.sum((ground.real -predict['output'][0:100]) ** 2 + (ground.imag - predict['output'][100:200]) ** 2) / ground.shape[0]
    losses.append(loss)

loss_array = np.array(losses)
choices = loss_array.argsort()
select = choice[choices[np.arange(-5, 5)]]
select.sort()

ground_truther = VN_coeff[select, ...]

predictions = classifier.predict(
    input_fn=lambda: fn.predict_hdf5_functor(transfer='reformed_spectra_densesapce_safe.hdf5',
                                             select=select,
                                             batch_size=1))

h5_reformed.close()

fig, ax = plt.subplots(nrows=int(ground_truther.shape[0]), ncols=2, figsize=(20, 170))
grid = np.indices(dimensions=(int(ground_truther.shape[0]), 1))
row = grid[0].flatten()
col = grid[1].flatten() * 2
index = np.arange(ground_truther.shape[0])
for ind, ro, co, predict in zip(index, row, col, predictions):
    ground_pulse = runningMan.VN_reconstructor(coeff=np.reshape(ground_truther[ind], newshape=(10, 10)),
                                               filename=VN_file)

    VN_predict = predict['output'][0:100] + 1J * predict['output'][100:200]
    predict_pulse = runningMan.VN_reconstructor(coeff=np.reshape(VN_predict, newshape=(10, 10)),
                                                filename=VN_file)

    cuts = (0, 1000)
    ax[ro, co].plot(ground_pulse[cuts[0]:cuts[1]].real, 'b', predict_pulse[cuts[0]:cuts[1]].real, 'r')
    np.savetxt('Images/ground_waveform{}_real'.format(ind),ground_pulse.real, delimiter=',')
    np.savetxt('Images/predict_waveform{}_real'.format(ind), predict_pulse.real, delimiter=',')

    ax[ro, co + 1].plot(ground_pulse[cuts[0]:cuts[1]].imag, 'b', predict_pulse[cuts[0]:cuts[1]].imag, 'r')
    np.savetxt('Images/ground_waveform{}_imag'.format(ind),ground_pulse.imag, delimiter=',')
    np.savetxt('Images/predict_waveform{}_imag'.format(ind), predict_pulse.imag, delimiter=',')

# display(fig)
# fig.savefig('Images/pulse_recon.png', dpi= 700)
