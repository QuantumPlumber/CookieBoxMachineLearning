import h5py
import numpy as np
import time
import multiprocessing as mp


def gaussian_kernel_compute_mp(energy_points, Spectra):
    # Convenience function to ensure destruction of intermediate large arrays.

    # reshape Spectra array before processing:
    local_spectra_shape = Spectra.shape
    #print(Spectra.shape)
    local_spectra_reshape = np.array(
        [local_spectra_shape[0] * local_spectra_shape[1], local_spectra_shape[2], local_spectra_shape[3]])
    spectra_reshaped = np.reshape(Spectra, newshape=local_spectra_reshape)

    #  use the old trick of setting all zero values to some outrageous number.
    #  No compute penalty as we already compute full array.
    #  Solves problem of zero padding at end of ragged arrays.
    energy_points_array = np.multiply.outer(np.ones_like(spectra_reshaped), energy_points)
    spectra_reshaped[spectra_reshaped == 0.0] = 1000
    # waveforms = np.zeros_like(spectra_reshaped)

    gaussian_centers = np.multiply.outer(spectra_reshaped, np.ones_like(energy_points))

    # Compute gaussian Kernel Density estimate over energy_points
    # waveforms must be summed over cookie
    waveforms = np.sum(np.exp(-(energy_points_array - gaussian_centers) ** 2 / (2 * .25 / 2.355)),
                       axis=2)

    # print(waveforms)
    return waveforms


def transform_2_spectra_from_mp(filename='../AttoStreakSimulations/TF_train_single.hdf5',
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
        Hits = h5file['Hits']
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

    chunksize = mp.cpu_count() // 3 * 2

    spectra_shape = np.array(Spectra.shape)
    # print(shape)
    spectra_reshape = np.array([spectra_shape[0] * spectra_shape[1], 16, num_ebins])
    if 'Spectra16' not in h5_reformed:
        detectors_ref = h5_reformed.create_dataset(name='Spectra16', compression='gzip', shape=spectra_reshape.tolist(),
                                                   chunks=(chunksize, spectra_reshape[1], spectra_reshape[2]))
    else:
        detectors_ref = h5_reformed['Spectra16']

    vn_shape = np.array(VN_coeff.shape)
    vn_reshape = np.array([spectra_reshape[0], vn_shape[1] * vn_shape[2]])
    if 'VN_coeff' not in h5_reformed:
        vn_co_ref = h5_reformed.create_dataset(name='VN_coeff', shape=vn_reshape.tolist(), compression='gzip',
                                               dtype='complex128',
                                               chunks=(chunksize, vn_reshape[1]))
    else:
        vn_co_ref = h5_reformed['VN_coeff']

    hits_shape = np.array(Hits.shape)
    hits_reshape = np.array([hits_shape[0] * hits_shape[1], hits_shape[2]])
    if 'Hits' not in h5_reformed:
        hits_ref = h5_reformed.create_dataset(name='Hits', shape=hits_reshape.tolist(), compression='gzip',
                                              chunks=(chunksize, hits_reshape[1]))
    else:
        hits_ref = h5_reformed['Hits']
    # hits_ref = np.reshape(Hits, hits_reshape)

    num_spectra = np.arange(0, spectra_shape[0], step=100)
    num_spectra_top = np.append(num_spectra, np.array(spectra_shape[0]) - 1)[1:]
    bot_top = np.column_stack((num_spectra, num_spectra_top)).astype('int')

    def data_generator(Hits, Spectra, VN_coeff, jump):
        for i in np.arange(0, spectra_shape[0], step=jump):
            yield Hits[i:i + jump, :, :], Spectra[i:i + jump, :, :, 1, 0:350], VN_coeff[i:i + jump, :, :], [i, i + jump]

    sim_data = data_generator(Hits=Hits, Spectra=Spectra, VN_coeff=VN_coeff, jump=chunksize)

    # Create Pool
    processes = chunksize
    pool = mp.Pool(processes)
    print('Pooled {} threads for parallel computation'.format(processes))

    #checkpoint_global = time.process_time()
    break_number = 0
    # Encapsulate memory heavy things in the slicer function so python is forced to free memory
    for hh, ss, vnvn, b_slice in sim_data:
        checkpoint = time.perf_counter()

        # reshape and record hits
        local_hits_shape = np.array(hh.shape)
        local_hits_reshape = np.array((local_hits_shape[0] * local_hits_shape[1], local_hits_shape[2]))
        reshaped_hits = np.reshape(hh, newshape=local_hits_reshape)
        hits_ref[(b_slice[0] * hits_shape[1]):(b_slice[1] * hits_shape[1]), :] = reshaped_hits

        # fill in the vn coefficients, requires copying the array.
        local_vn_shape = np.array(vnvn.shape)
        local_vn_reshape = np.array([local_vn_shape[0], local_vn_shape[1] * local_vn_shape[2]])
        vn_reshped = np.repeat(
            np.reshape(vnvn, newshape=local_vn_reshape), repeats=hits_shape[1], axis=0)
        vn_co_ref[(b_slice[0] * hits_shape[1]):(b_slice[1] * hits_shape[1]), :] = vn_reshped

        # Create 16 detector lists, discards angle information.
        workers = []
        for spect in ss:
            argslist = (energy_points, np.expand_dims(spect, axis=0))
            #print(argslist[1].shape)
            worker = pool.apply_async(gaussian_kernel_compute_mp, argslist)
            workers.append(worker)

        # transformed_spectra = gaussian_kernel_compute_mp(energy_points=energy_points, Spectra=ss)
        transformed_spectra_list = []
        for worker in workers:
            transformed_spectra_list.append(worker.get())
        #print(transformed_spectra_list[0].shape)
        transformed_spectra = np.concatenate(transformed_spectra_list, axis=0)
        #print(transformed_spectra.shape)
        detectors_ref[(b_slice[0] * hits_shape[1]):(b_slice[1] * hits_shape[1]), :, :] = transformed_spectra

        print('complete {} to {} of {}'.format(b_slice[0], b_slice[1], spectra_shape[0]))

        delta_t = checkpoint - time.perf_counter()
        print('Converted in {}'.format(delta_t))
        if break_number == 100000:
            break
        break_number += 1

    #delta_t = checkpoint_global - time.process_time()
    #print('Total Runtime was {}'.format(delta_t))

    h5file.close()
    h5_reformed.close()
