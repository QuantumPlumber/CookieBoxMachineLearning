'''
This file contains functions to:
import data
create TF datasets

live large

'''

import tensorflow as tf
import h5py
import numpy as np


def transform(filename='CookieBox/AttoStreakSimulations/TF_train_single_pulse.hdf5'):
    '''
    Transforms the raw simulation data into detector data of a given precision

    :param filename:
    :return:
    '''

    h5file = h5py.File(filename, 'r')

    if 'Hits' in h5file:
        Hits = h5file['Hits'][...]
    else:
        raise Exception('No "Hits" in file.')

    if 'Spectra' in h5file:
        Spectra = h5file['Spectra'][...]
    else:
        raise Exception('No "Spectra" in file.')

    if 'VN_coeff' in h5file:
        VN_coeff = h5file['VN_coeff'][...]
    else:
        raise Exception('No "VN_coeff" in file.')

    shape = np.array(Spectra.shape)
    print(shape)
    shape[2] = 16
    detectors = np.empty(shape)

    # Create 16 detector lists, discards angle information.
    detector_thetas = np.arange(-np.pi, np.pi, 2 * np.pi / 16)
    for angle in detector_thetas:
        positions = (np.abs(Spectra[:, :, 0, :] - angle) < np.abs(detector_thetas[0] - detector_thetas[1]) / 2.)

        loc = np.where(detector_thetas == angle)[0]

        cut = (Spectra[:, :, 1, :])[positions]

        detectors[:, :, loc, :] = cut

    d1_shape = detectors.shape[0] * detectors.shape[1]
    hits_out = np.reshape(detectors, newshape=d1_shape)
    spectra_out = np.reshape(detectors, newshape=(d1_shape, detectors.shape[2], detectors.shape[2]))

    h5file.close()

    return hits_out, spectra_out, VN_coeff


def transform_2_TFRecord(filename='../AttoStreakSimulations/TF_train_single_pulse.hdf5', TFRecord = 'TF_train_single_pulse.tfrecord' ):
    '''
    Transforms the raw simulation data into TFRecord data format

    :param filename:
    :return:
    '''

    h5file = h5py.File(filename, 'r')

    if 'Hits' in h5file:
        Hits = h5file['Hits'][0:10,...]
    else:
        raise Exception('No "Hits" in file.')

    if 'Spectra' in h5file:
        Spectra = h5file['Spectra'][0:10,...]
    else:
        raise Exception('No "Spectra" in file.')

    if 'VN_coeff' in h5file:
        VN_coeff = h5file['VN_coeff'][0:10,...]
    else:
        raise Exception('No "VN_coeff" in file.')
    '''
    #  reshape arrays and load into memory 
    Hits = Hits[...].flatten()

    examples = Hits.shape[0]
    shape = np.array(Spectra.shape)
    newshape = np.array([examples, shape[2], shape[3]])

    Spectra = np.reshape(Spectra[...], newshape=newshape)

    newshape = np.array([examples, VN_coeff[1] * VN_coeff[2]])
    VN_coeff = np.reshape(VN_coeff, newshape=newshape)
    '''
    # Create 16 detector lists, discards angle information.
    detector_thetas = np.arange(-np.pi, np.pi, 2 * np.pi / 16)
    det_names = np.empty([1])
    for n in range(detector_thetas):
        det_names += 'det{}'.format(n)
    print(det_names)
    flist = np.empty(detector_thetas.shape)

    # context manager for file writer
    with tf.python_io.TFRecordWriter(TFRecord) as writer:
        # loop through each pulse shape
        for spectral in np.arange(Spectra.shape[0]):
            hits = Hits[spectral]
            spect = Spectra[spectral]
            vn_coeff = VN_coeff[spectral, ...].flatten()

            # loop through each random sample
            for spec in range(hits.shape[0]):
                sp = hits[spec, ...]

                # separate into 16 detector lists
                for angle in range(detector_thetas):
                    positions = (np.abs(sp[0, :] - detector_thetas[angle]) < np.abs(
                        detector_thetas[0] - detector_thetas[1]) / 2.)

                    flist[angle] = tf.train.Feature(byteslist=tf.train.FloatList(value=(sp[1, :])[positions]))

                spec_list = tf.train.FeatureList(flist)
                vn_list = tf.train.FeatureList(
                    (tf.train.Feature(byteslist=tf.train.FloatList(value=vn_coeff))))  # Single element list
                hits_list = tf.train.FeatureList(
                    (tf.train.Feature(byteslist=tf.train.FloatList(value=hits))))  # Single element list, single element

                feature_dict = {'Det_Spect': spec_list,
                                'VN_Coeff': vn_list,
                                'num_hits': hits_list
                                }
                feat_lists =tf.train.FeatureLists(feature_list=feature_dict)

                seq_example = tf.train.SequenceExample(feature_lists=feat_lists)
                writer.write(seq_example.SerializeToString())

    h5file.close()

    return

