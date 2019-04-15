import h5py
import numpy as np
import time
import multiprocessing as mp


def gaussian_kernel_compute_mp(energy_points, Spectra):
    # Convenience function to ensure destruction of intermediate large arrays.

    # reshape Spectra array before processing:
    local_spectra_shape = Spectra.shape
    # print(Spectra.shape)
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

    if 'Time_pulse' in h5file:
        Time_pulse = h5file['Time_pulse']
    else:
        raise Exception('No "Time_pulse" in file.')

    num_ebins = 100
    energy_range = 100.  # 100 eV was used in the code
    energy_points = np.linspace(0, energy_range, num_ebins)

    chunksize = mp.cpu_count()

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

    pulse_shape = np.array(Time_pulse.shape)
    pulse_reshape = np.array([spectra_reshape[0], 2, int(pulse_shape[2] / 10)])  # reshaped for averaging
    if 'Pulse_truth' not in h5_reformed:
        Pulse_truth = h5_reformed.create_dataset(name='Pulse_truth', shape=pulse_reshape.tolist(), compression='gzip',
                                                 chunks=(chunksize, pulse_reshape[1], pulse_reshape[2]))
    else:
        Pulse_truth = h5_reformed['Pulse_truth']

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
            yield Hits[i:i + jump, :, :], Spectra[i:i + jump, :, :, 1, 0:350], Time_pulse[i:i + jump, (0, 1),
                                                                               :], VN_coeff[i:i + jump, :, :], [i,
                                                                                                                i + jump]

    sim_data = data_generator(Hits=Hits, Spectra=Spectra, VN_coeff=VN_coeff, jump=chunksize)

    # Create Pool
    processes = chunksize
    pool = mp.Pool(processes)
    print('Pooled {} threads for parallel computation'.format(processes))

    # checkpoint_global = time.process_time()
    break_number = 0
    # Encapsulate memory heavy things in the slicer function so python is forced to free memory
    for hh, ss, pulsepulse, vnvn, b_slice in sim_data:
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

        # fill in the pulse_mag and pulse_phase, requires copying the array.
        # local_pulse_shape = np.array(pulsepulse.shape)
        take_every_10 = np.arange(start=0, stop=1000, step=10)
        phase_diff = np.append(np.zeros(shape=(pulsepulse.shape[0], 1, 1)),
                               pulsepulse[:, 1:, 1:] - pulsepulse[:, 1:, :-1], axis=2)
        phase_diff[phase_diff < 0] = phase_diff[phase_diff < 0] + 2 * np.pi
        cumulative_phase = np.concatenate(
            (pulsepulse[:, 0:1, take_every_10], np.cumsum(phase_diff, axis=2)[:, :, take_every_10]), axis=1)
        pulsepulse_repeat = np.repeat(cumulative_phase, repeats=hits_shape[1], axis=0)
        Pulse_truth[(b_slice[0] * hits_shape[1]):(b_slice[1] * hits_shape[1]), :, :] = pulsepulse_repeat

        # Create 16 detector lists, discards angle information.
        workers = []
        for spect in ss:
            argslist = (energy_points, np.expand_dims(spect, axis=0))
            # print(argslist[1].shape)
            worker = pool.apply_async(gaussian_kernel_compute_mp, argslist)
            workers.append(worker)

        # transformed_spectra = gaussian_kernel_compute_mp(energy_points=energy_points, Spectra=ss)
        transformed_spectra_list = []
        for worker in workers:
            transformed_spectra_list.append(worker.get())
        # print(transformed_spectra_list[0].shape)
        transformed_spectra = np.concatenate(transformed_spectra_list, axis=0)
        # print(transformed_spectra.shape)
        detectors_ref[(b_slice[0] * hits_shape[1]):(b_slice[1] * hits_shape[1]), :, :] = transformed_spectra

        print('complete {} to {} of {}'.format(b_slice[0], b_slice[1], spectra_shape[0]))

        delta_t = checkpoint - time.perf_counter()
        print('Converted in {}'.format(delta_t))
        if break_number == 1000000:
            break
        break_number += 1

    pool.close()  # tell the pool no more processes will be submitted
    pool.join()  # wait for the pool to complete all computations before calling results

    # delta_t = checkpoint_global - time.process_time()
    # print('Total Runtime was {}'.format(delta_t))

    h5file.close()
    h5_reformed.close()


def pulse_evaluate(filename='../AttoStreakSimulations/TF_train_single.hdf5',
                   transfer='reformed_spectra.hdf5',
                   num_spectra=100000):
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

    if 'Spectra16' in h5file:
        Spectra16 = h5file['Spectra16']
    else:
        raise Exception('No "Spectra16" in file.')

    if 'VN_coeff' in h5file:
        VN_coeff = h5file['VN_coeff']
    else:
        raise Exception('No "VN_coeff" in file.')

    if 'Pulse_truth' in h5file:
        Pulse_truth = h5file['Pulse_truth']
    else:
        raise Exception('No "Pulse_truth" in file.')

    num_ebins = 100
    energy_range = 100.  # 100 eV was used in the code
    energy_points = np.linspace(0, energy_range, num_ebins)

    chunksize = mp.cpu_count()

    if 'Spectra16' not in h5_reformed:
        detectors_ref = h5_reformed.create_dataset(name='Spectra16',
                                                   compression='gzip',
                                                   shape=(num_spectra, Spectra16.shape[1], Spectra16.shape[2]),
                                                   chunks=(chunksize, Spectra16.shape[1], Spectra16.shape[2]))
    else:
        detectors_ref = h5_reformed['Spectra16']

    if 'Pulse_truth' not in h5_reformed:
        Pulse_truth_ref = h5_reformed.create_dataset(name='Pulse_truth',
                                                     shape=(num_spectra, Pulse_truth.shape[1], Pulse_truth.shape[2]),
                                                     compression='gzip',
                                                     chunks=(chunksize, Pulse_truth.shape[1], Pulse_truth.shape[2]))
    else:
        Pulse_truth_ref = h5_reformed['Pulse_truth']

    def data_generator(Spectra, pulse_truth, jump):
        spectra_index = np.arange(Spectra.shape[0])
        while True:
            pulse_number = np.random.randint(1, 5, size=jump)
            #pulse_number = np.repeat(2, repeats=jump)
            pulses = []
            for num in pulse_number:
                pulses.append(np.sort(np.random.choice(spectra_index, size=num, replace=False), axis=0))
            # print(pulses)
            spectra_out_list = []
            pulse_truth_out_list = []
            for num, group in zip(pulse_number, pulses):
                # print(Spectra[group.tolist(), ...].shape)
                spectra_out_list.append(np.sum(Spectra[group.tolist(), ...], axis=0) / float(num))
                pulse_truth_out_list.append(np.sum(pulse_truth[group.tolist(), ...], axis=0) / float(num))

            yield [np.stack(spectra_out_list, axis=0), np.stack(pulse_truth_out_list, axis=0)]

    sim_data = data_generator(Spectra=Spectra16, pulse_truth=Pulse_truth, jump=chunksize)

    def chunk_maker(Spectra, jump):
        for i in np.arange(0, Spectra.shape[0], step=jump):
            yield [i, i + jump]

    chunks = chunk_maker(Spectra=detectors_ref, jump=chunksize)

    # checkpoint_global = time.process_time()
    break_number = 0
    # Encapsulate memory heavy things in the slicer function so python is forced to free memory
    for chunk, pulsepulsetrutru in zip(chunks, sim_data):

        if break_number % 1001 == 0:
            chunk_start = chunk[0]
            checkpoint = time.perf_counter()

        detectors_ref[chunk[0]:chunk[1], ...] = pulsepulsetrutru[0]
        Pulse_truth_ref[chunk[0]:chunk[1], ...] = pulsepulsetrutru[1]

        if break_number % 1000 == 0:
            delta_t = checkpoint - time.perf_counter()
            print('Scrambled {} waveforms in {}'.format(chunk[1] - chunk_start, delta_t))

        if break_number == int(1e6):
            break
        break_number += 1
        # print(break_number)

    # delta_t = checkpoint_global - time.process_time()
    # print('Total Runtime was {}'.format(delta_t))

    h5file.close()
    h5_reformed.close()
