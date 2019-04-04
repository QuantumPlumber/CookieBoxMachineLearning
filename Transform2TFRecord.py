import tensorflow as tf
import h5py
import numpy as np
import time


def reformat_2_TFRecord(transfer='dummy.hdf5', TFRecord='test_TFRecord', events_per_file=1000):
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

    if 'VN_coeff' not in h5_reformed:
        raise Exception('No "VN_coeff" in file.')
    else:
        VN_coeff = h5_reformed['VN_coeff']

    print(h5_reformed.keys())
    for key in list(h5_reformed.keys()):
        print('shape of {} is {}'.format(key, h5_reformed[key].shape))

    random_shuffled_index = np.arange(0, Spectra16.shape[0])
    np.random.shuffle(random_shuffled_index)

    def file_chunker(start, stop, step, base_filename):
        for ii in np.arange(start, stop, step):
            yield [ii, ii + step], '{}_{}-{}'.format(base_filename, ii, ii + step)

    file_chunks = file_chunker(start=0, stop=random_shuffled_index.shape[0], step=events_per_file,
                               base_filename=TFRecord)
    j = 0
    for range, filename in file_chunks:
        # context manager for file writer
        with tf.python_io.TFRecordWriter(filename) as writer:
            # loop through each random sample
            i = 0
            checkpoint = time.perf_counter()
            for event in random_shuffled_index[range[0]:range[1]]:
                i = i + 1

                spectral = Spectra16[event, ...]
                VN_coeff_select = VN_coeff[event, ...]
                vn_coeff = np.concatenate((np.abs(VN_coeff_select), np.angle(VN_coeff_select)), axis=0)

                cookies = []
                for spec in spectral:
                    # separate into 16 detector lists

                    cookies.append(tf.train.Feature(float_list=tf.train.FloatList(value=spec)))

                spec_list = tf.train.FeatureList(feature=cookies)

                vn_feature = tf.train.Features(feature={
                    'VN_coeff': tf.train.Feature(
                        float_list=tf.train.FloatList(value=vn_coeff))})  # Single element list

                vn_features = tf.train.Features

                feature_dict = {'spectra': spec_list}
                feat_lists = tf.train.FeatureLists(feature_list=feature_dict)

                seq_example = tf.train.SequenceExample(context=vn_feature, feature_lists=feat_lists)
                writer.write(seq_example.SerializeToString())

                if (i % 1000) == 1:
                    delta_t = checkpoint - time.perf_counter()
                    print('Converted in {}'.format(delta_t))
                    checkpoint = time.perf_counter()

        print('completed file {}'.format(j))
        if j == 10000:
            break
        j = j + 1


    h5_reformed.close()

    return
