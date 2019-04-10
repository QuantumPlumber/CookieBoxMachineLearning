import h5py
import numpy as np
import matplotlib.pyplot as plt

h5_file = h5py.File('convert_test.hdf5', 'r')
print(h5_file.keys())
for key in list(h5_file.keys()):
    print('shape of {} is {}'.format(key, h5_file[key].shape))

select = np.sort(np.random.random_integers(0, 100, 5))
#select = np.arange(5)
pulse = h5_file['Pulse_truth'][select, :, :]

h5_file.close()

fig, ax = plt.subplots(nrows=int(pulse.shape[0]), ncols=int(pulse.shape[1]), figsize=(22, 17),
                       sharex=True)
grid = np.indices(dimensions=(int(pulse.shape[0]), int(pulse.shape[1])))
row = grid[0].flatten()
col = grid[1].flatten()
index = np.arange(pulse.shape[0])

for ro, co in zip(row, col):
    ax[ro, co].plot(pulse[ro, co, :])

# fig.savefig('Images/KernelDensityEstimate.png', dpi=700)
