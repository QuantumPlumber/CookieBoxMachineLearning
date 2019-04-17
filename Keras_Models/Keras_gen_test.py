import numpy as np
import h5py


def data_generator(transfer='TF_train_wave_unwrapped_eggs.hdf5', batch_size=64, cut_bot=.8, cut_top=1.):
    '''
        Reformats transformed simulation data into TFRecord data format

        :param filename:
        :return:
        '''

    h5_reformed = h5py.File(transfer, 'r')

    if 'Spectra16' not in h5_reformed:
        raise Exception('No "Spectra16" in file.')
    else:
        Spectra16 = h5_reformed['Spectra16']

    if 'Pulse_truth' not in h5_reformed:
        raise Exception('No "Pulse_truth" in file.')
    else:
        Pulse_truth = h5_reformed['Pulse_truth']

    random_shuffled_index = np.arange(0, Spectra16.shape[0])[
                            int(Spectra16.shape[0] * cut_bot):int(Spectra16.shape[0] * cut_top)]
    np.random.shuffle(random_shuffled_index)

    print((int(Spectra16.shape[0] * cut_bot), int(Spectra16.shape[0] * cut_top)))

    for batch in np.arange(start=0, stop=random_shuffled_index.shape[0], step=batch_size):
        print(batch)
        print(np.sort(random_shuffled_index[batch: batch + batch_size]))
        return (Spectra16[np.sort(random_shuffled_index[batch: batch + batch_size]), ...],
                Pulse_truth[np.sort(random_shuffled_index[batch: batch + batch_size]), 0, :],
                Pulse_truth[np.sort(random_shuffled_index[batch: batch + batch_size]), 1, :])
        break


train_data = data_generator(transfer='../TF_train_wave_unwrapped_eggs.hdf5', batch_size=64, cut_bot=.0, cut_top=.8)

for train in train_data:
    print(train[0].shape)
    break
