'''
This file contains functions to:
import data
create TF datasets

live large

'''

import tensorflow as tf
import h5py
import numpy as np
import time
import importlib as imp


def transform(filename='../AttoStreakSimulations/TF_train_single_pulse.hdf5', transfer='reformed.hdf5'):
    '''
    Transforms the raw simulation data into detector data of a given precision

    Combines the first and second dimension for easier shuffling.
    :param filename:
    :return:
    '''

    h5file = h5py.File(filename, 'r')
    h5_reformed = h5py.File(transfer)

    if 'Hits' in h5file:
        Hits = h5file['Hits'][...]
    else:
        raise Exception('No "Hits" in file.')

    if 'Spectra' in h5file:
        Spectra = h5file['Spectra']
    else:
        raise Exception('No "Spectra" in file.')

    if 'VN_coeff' in h5file:
        VN_coeff = h5file['VN_coeff']
    else:
        raise Exception('No "VN_coeff" in file.')

    spectra_shape = np.array(Spectra.shape)
    # print(shape)
    spectra_reshape = np.array([spectra_shape[0] * spectra_shape[1], 16, spectra_shape[3]])
    if 'Spectra16' not in h5_reformed:
        detectors_ref = h5_reformed.create_dataset(name='Spectra16', compression='gzip', shape=spectra_reshape.tolist())
    else:
        detectors_ref = h5_reformed['Spectra16']

    vn_shape = np.array(VN_coeff.shape)
    vn_reshape = np.append(spectra_reshape[0], vn_shape[1:])
    if 'VN_coeff' not in h5_reformed:
        vn_co_ref = h5_reformed.create_dataset(name='VN_coeff', shape=vn_reshape.tolist(), compression='gzip',
                                               dtype='complex128')
    else:
        vn_co_ref = h5_reformed['VN_coeff']

    hits_shape = np.array(Hits.shape)
    hits_reshape = hits_shape[0] * hits_shape[1]
    if 'Hits' not in h5_reformed:
        hits_ref = h5_reformed.create_dataset(name='Hits', shape=(hits_reshape.tolist(),))
    else:
        hits_ref = h5_reformed['Hits']
    hits_ref = np.reshape(Hits, hits_reshape)

    num_spectra = np.arange(0, spectra_shape[0], step=10)
    num_spectra_top = np.append(num_spectra, np.array(spectra_shape[0]) - 1)[1:]
    bot_top = np.column_stack((num_spectra, num_spectra_top)).astype('int')

    detector_thetas = np.arange(-np.pi, np.pi, 2 * np.pi / 16)
    enum = np.arange(detector_thetas.shape[0])
    enum_theta = np.column_stack((enum, detector_thetas))
    '''
    for bot, top in bot_top:
        # Create 16 detector lists, discards angle information.
        detector = np.zeros(spectra_reshape)[bot,top]
        for loc, angle in enum_theta:
            positions[...] = (np.abs(Spectra[bot:top, :, 0, :] - angle) < np.abs(
                detector_thetas[0] - detector_thetas[1]) / 2.)
            
            vals = (Spectra[bot:top, :, 1, :])[positions]
            detector[:,loc,:] = (Spectra[bot:top, :, 1, :])[positions]

        #coeff = VN_coeff[bot:top]
        #vn_co_ref[bot*spectra_shape[1]:top*spectra_shape[1]] = np.repeat(coeff, repeats=spectra_shape[1], axis=0)
        #detectors_ref[bot*spectra_shape[1]:top*spectra_shape[1],:,:] = detector
    '''
    # Encapsulate memory heavy things in the slicer function so python is forced to free memory
    for bot, top in bot_top[...]:
        # Create 16 detector lists, discards angle information.
        slicer(bot, top, spectra_shape, spectra_reshape, enum_theta, detector_thetas, Spectra, VN_coeff, vn_co_ref,
               detectors_ref)
        print('complete {} to {} of {}'.format(bot, top, spectra_reshape[0]))
    h5file.close()
    h5_reformed.close()


def slicer(bot, top, spectra_shape, spectra_reshape, enum_theta, detector_thetas, Spectra, VN_coeff, vn_co_ref,
           detectors_ref):
    detector = np.zeros((top - bot, spectra_shape[1], 16, spectra_shape[3]))
    for loc, angle in enumerate(detector_thetas.tolist()):
        positions = (np.abs(Spectra[bot:top, :, 0, :] - angle) < np.abs(
            detector_thetas[0] - detector_thetas[1]) / 2.)
        (detector[:, :, loc, :])[positions] = (Spectra[bot:top, :, 1, :])[positions]

    coeff = np.repeat(VN_coeff[bot:top], repeats=spectra_shape[1], axis=0)
    cut_bot = bot * spectra_shape[1]
    cut_top = top * spectra_shape[1]
    vn_co_ref[cut_bot:cut_top] = coeff
    detectors_ref[bot * spectra_shape[1]:top * spectra_shape[1], :, :] = np.reshape(detector, newshape=(
        (top - bot) * spectra_shape[1], 16, spectra_shape[3]))


def transform_2_spectra(filename='../AttoStreakSimulations/TF_train_single.hdf5',
                        transfer='reformed_spectra.hdf5'):
    '''
    Transforms the raw simulation data into detector data of a given precision

    Combines the first and second dimension for easier shuffling.
    :param filename:
    :return:
    '''

    h5file = h5py.File(filename, 'r')
    h5_reformed = h5py.File(transfer)

    if 'Hits' in h5file:
        Hits = h5file['Hits'][...]
    else:
        raise Exception('No "Hits" in file.')

    if 'Spectra' in h5file:
        Spectra = h5file['Spectra']
    else:
        raise Exception('No "Spectra" in file.')

    if 'VN_coeff' in h5file:
        VN_coeff = h5file['VN_coeff']
    else:
        raise Exception('No "VN_coeff" in file.')

    num_ebins = 100
    energy_range = 100.  # 100 eV was used in the code
    energy_points = np.linspace(0, energy_range, num_ebins)

    spectra_shape = np.array(Spectra.shape)
    # print(shape)
    spectra_reshape = np.array([spectra_shape[0] * spectra_shape[1], 16, num_ebins])
    if 'Spectra16' not in h5_reformed:
        detectors_ref = h5_reformed.create_dataset(name='Spectra16', compression='gzip', shape=spectra_reshape.tolist())
    else:
        detectors_ref = h5_reformed['Spectra16']

    vn_shape = np.array(VN_coeff.shape)
    vn_reshape = np.append(spectra_reshape[0], vn_shape[1] * vn_shape[2])
    if 'VN_coeff' not in h5_reformed:
        vn_co_ref = h5_reformed.create_dataset(name='VN_coeff', shape=vn_reshape.tolist(), compression='gzip',
                                               dtype='complex128')
    else:
        vn_co_ref = h5_reformed['VN_coeff']

    hits_shape = np.array(Hits.shape)
    hits_reshape = hits_shape[0] * hits_shape[1]
    if 'Hits' not in h5_reformed:
        hits_ref = h5_reformed.create_dataset(name='Hits', shape=(hits_reshape.tolist(),))
    else:
        hits_ref = h5_reformed['Hits']
    hits_ref = np.reshape(Hits, hits_reshape)

    num_spectra = np.arange(0, spectra_shape[0], step=100)
    num_spectra_top = np.append(num_spectra, np.array(spectra_shape[0]) - 1)[1:]
    bot_top = np.column_stack((num_spectra, num_spectra_top)).astype('int')

    detector_thetas = np.arange(-np.pi, np.pi, 2 * np.pi / 16)
    enum = np.arange(detector_thetas.shape[0])
    enum_theta = np.column_stack((enum, detector_thetas))

    # Encapsulate memory heavy things in the slicer function so python is forced to free memory
    for bot, top in bot_top[...]:
        checkpoint = time.process_time()
        # Create 16 detector lists, discards angle information.
        slice_2_spectra(bot, top, spectra_shape, spectra_reshape, vn_reshape, energy_points, enum_theta,
                        detector_thetas, Spectra, VN_coeff, vn_co_ref, detectors_ref)
        print('complete {} to {} of {}'.format(bot, top, spectra_shape[0]))
        delta_t = checkpoint - time.process_time()
        print('Converted in {}'.format(delta_t))
    h5file.close()
    h5_reformed.close()


def slice_2_spectra(bot, top, spectra_shape, spectra_reshape, vn_reshape, energy_points, enum_theta, detector_thetas,
                    Spectra, VN_coeff, vn_co_ref, detectors_ref):
    # Compute the kernel density estimate over the energy_points of the waveform.
    cut_shape = (top - bot, spectra_shape[1], spectra_shape[3])
    waveforms = gaussian_kernel_compute(cut_shape, bot, top, energy_points, Spectra)

    detector = slice_waveform(bot, top, spectra_shape, spectra_reshape, energy_points, detector_thetas, Spectra,
                              waveforms)

    coeff = np.repeat(VN_coeff[bot:top], repeats=spectra_shape[1], axis=0)
    cut_bot = bot * spectra_shape[1]
    cut_top = top * spectra_shape[1]
    vn_co_ref[cut_bot:cut_top, :] = np.reshape(coeff, newshape=(cut_top - cut_bot, vn_reshape[1]))
    detectors_ref[cut_bot:cut_top, :, :] = np.reshape(detector, newshape=(
        cut_top - cut_bot, spectra_reshape[1], spectra_reshape[2]))


def gaussian_kernel_compute(cut_shape, bot, top, energy_points, Spectra):
    # Convenience function to ensure destruction of intermediate large arrays.

    #  use the old trick of setting all zero values to some outrageous number.
    #  No compute penalty as we already compute full array.
    #  Solves problem of zero padding at end of ragged arrays.
    energy_points_array = np.multiply.outer(np.ones(cut_shape), energy_points)
    spectra_rem = Spectra[bot:top, :, 1, :]
    spectra_rem[spectra_rem == 0.0] = 1000
    gaussian_centers = np.multiply.outer(spectra_rem, np.ones_like(energy_points))

    waveforms = np.exp(
        -(energy_points_array - gaussian_centers) ** 2 / (2 * .25 / 2.355))  # waveforms must be summed over cookie

    # print(waveforms)
    return waveforms


def slice_waveform(bot, top, spectra_shape, spectra_reshape, energy_points, detector_thetas, Spectra, waveforms):
    #  another convenience function to encapsulate the largest arrays.

    detector = np.zeros((top - bot, spectra_shape[1], 16, spectra_shape[3], energy_points.shape[0]))
    # print(bot, top)
    # print(detector.shape)
    # print(detector.nbytes)
    for loc, angle in enumerate(detector_thetas.tolist()):
        # contracts along the third axis to form
        angle_index = np.multiply.outer(np.abs(Spectra[bot:top, :, 0, :] - angle) < np.abs(
            detector_thetas[0] - detector_thetas[1]) / 2., np.ones_like(energy_points)).astype('bool')
        # print(angle_index)
        # angle information is encoded in the fourth, energy waveform in the 5th.
        (detector[:, :, loc, :, :])[angle_index] = waveforms[angle_index]
    return np.sum(detector, axis=3)


def transform_shuffle(filename='../AttoStreakSimulations/TF_train_single_pulse.hdf5', transfer='reformed.hdf5'):
    '''
    Transforms the raw simulation data into detector data of a given precision

    Combines the first and second dimension for easier shuffling.
    :param filename:
    :return:
    '''

    h5file = h5py.File(filename, 'r')
    h5_reformed = h5py.File(transfer)

    if 'Hits' in h5file:
        Hits = h5file['Hits'][...]
    else:
        raise Exception('No "Hits" in file.')

    if 'Spectra' in h5file:
        Spectra = h5file['Spectra']
    else:
        raise Exception('No "Spectra" in file.')

    if 'VN_coeff' in h5file:
        VN_coeff = h5file['VN_coeff']
    else:
        raise Exception('No "VN_coeff" in file.')

    spectra_shape = np.array(Spectra.shape)
    # print(shape)
    spectra_reshape = np.array([spectra_shape[0] * spectra_shape[1], 16, spectra_shape[3]])
    print(spectra_reshape)
    if 'Spectra16' not in h5_reformed:
        detectors_ref = h5_reformed.create_dataset(name='Spectra16', shape=spectra_reshape.tolist())

    vn_shape = np.array(VN_coeff.shape)
    vn_reshape = np.append(spectra_reshape[0], vn_shape[1:])
    if 'VN_coeff' not in h5_reformed:
        vn_co_ref = h5_reformed.create_dataset(name='VN_coeff', shape=vn_reshape.tolist())

    hits_shape = np.array(Hits.shape)
    hits_reshape = hits_shape[0] * hits_shape[1]
    if 'Hits' not in h5_reformed:
        hits_ref = h5_reformed.create_dataset(name='Hits', shape=(hits_reshape.tolist(),))

    # Generate slicing arrays
    row, column = np.indices(spectra_shape[0:2])
    shuffle = np.random.permutation(spectra_shape[0] * spectra_shape[1])
    row = row.flatten()[shuffle]
    column = column.flatten()[shuffle]

    num_spectra = np.arange(0, row.shape[0], step=row.shape[0] / 100).astype('int')
    num_spectra_top = np.append(num_spectra, row.shape[0])[1:]
    bot_top = np.column_stack((num_spectra, num_spectra_top))

    detector_thetas = np.arange(-np.pi, np.pi, 2 * np.pi / 16)
    enum = np.arange(detector_thetas.shape[0])
    enum_theta = np.column_stack((enum, detector_thetas))

    detector = np.zeros(spectra_reshape)[bot_top[0, 1]:bot_top[0, 0]]
    positions = np.zeros_like(detector).astype('bool')
    for bot, top in bot_top[0:1]:
        # Create 16 detector lists, discards angle information.
        print(bot, top)
        for loc, angle in enum_theta:
            print(row[bot:top].sort())
            print(column[bot:top].sort())
            positions[...] = (np.abs(Spectra[row[bot:top].sort(), column[bot:top].sort, 0, :] - angle) < np.abs(
                detector_thetas[0] - detector_thetas[1]) / 2.)

            detector[:, loc, 0:positions.shape[-1]] = (Spectra[bot:top, :, 1, :])[positions]

        hits_ref[bot * spectra_shape[1]:top * spectra_shape[1]] = Hits[row[bot:top], column[bot:top]]
        vn_co_ref[bot * spectra_shape[1]:top * spectra_shape[1]] = VN_coeff[row[bot:top]]
        detectors_ref[bot * spectra_shape[1]:top * spectra_shape[1], :, :] = detector

    h5file.close()
    h5_reformed.close()


def transform_2_TFRecord(filename='../AttoStreakSimulations/TF_train_single_pulse.hdf5',
                         TFRecord='TF_train_single_pulse.tfrecord'):
    '''
    Transforms the raw simulation data into TFRecord data format

    :param filename:
    :return:
    '''

    h5file = h5py.File(filename, 'r')

    if 'Hits' in h5file:
        Hits = h5file['Hits'][0:1, ...]
    else:
        raise Exception('No "Hits" in file.')

    if 'Spectra' in h5file:
        Spectra = h5file['Spectra'][0:1, ...]
    else:
        raise Exception('No "Spectra" in file.')

    if 'VN_coeff' in h5file:
        VN_coeff = h5file['VN_coeff'][0:1, ...]
    else:
        raise Exception('No "VN_coeff" in file.')

    # Create 16 detector lists, discards angle information
    detector_thetas = np.arange(-np.pi, np.pi, 2 * np.pi / 16)
    flist = []
    print(Spectra.shape)

    checkpoint = time.process_time()
    # context manager for file writer
    with tf.python_io.TFRecordWriter(TFRecord) as writer:
        # loop through each pulse shape
        for spectral in np.arange(Spectra.shape[0]):
            hits = Hits[spectral]
            spect = Spectra[spectral, ...]
            vn_coeff = VN_coeff[spectral, ...].flatten()

            # loop through each random sample
            for spec in np.arange(spect.shape[0]):
                sp = spect[spec, ...]
                # separate into 16 detector lists
                for angle in range(detector_thetas.shape[0]):
                    positions = (np.abs(sp[0, :] - detector_thetas[angle]) < np.abs(
                        detector_thetas[0] - detector_thetas[1]) / 2.)

                    flist.append(tf.train.Feature(float_list=tf.train.FloatList(value=(sp[1, :])[positions])))

                spec_list = tf.train.FeatureList(feature=flist)

                vn_list_real = tf.train.FeatureList(
                    feature=[
                        tf.train.Feature(float_list=tf.train.FloatList(value=vn_coeff.real))])  # Single element list

                vn_list_imag = tf.train.FeatureList(
                    feature=[
                        tf.train.Feature(float_list=tf.train.FloatList(value=vn_coeff.imag))])  # Single element list

                hits_list = tf.train.FeatureList(feature=[
                    tf.train.Feature(float_list=tf.train.FloatList(value=hits))])  # Single element list, single element

                feature_dict = {'Det_Spect': spec_list,
                                'VN_Coeff_real': vn_list_real,
                                'VN_Coeff_imag': vn_list_imag,
                                'num_hits': hits_list
                                }
                feat_lists = tf.train.FeatureLists(feature_list=feature_dict)

                seq_example = tf.train.SequenceExample(feature_lists=feat_lists)
                writer.write(seq_example.SerializeToString())

    delta_t = checkpoint - time.process_time()
    print('Converted in {}'.format(delta_t))

    h5file.close()

    return
