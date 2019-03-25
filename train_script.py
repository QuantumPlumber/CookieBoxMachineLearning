import numpy as np

'''
classifier.train(
    # input_fn=lambda: fn.input_functor(datanorm=testdatanorm, labelsnorm=testlabelsnorm, batch_size=1),
    input_fn=lambda: fn.input_hdf5_functor(transfer='reformed_spectra_densesapce_safe.hdf5', select=(0, 6000), batch_size=1),
    steps=6000)

classifier.train(
    #input_fn=lambda: fn.input_functor(datanorm=testdatanorm, labelsnorm=testlabelsnorm, batch_size=1),
    input_fn=lambda: fn.input_hdf5_functor(transfer='reformed_spectra_densesapce_safe.hdf5', select=(6000, 12000), batch_size=1),
    steps=6000)

classifier.train(
    #input_fn=lambda: fn.input_functor(datanorm=testdatanorm, labelsnorm=testlabelsnorm, batch_size=1),
    input_fn=lambda: fn.input_hdf5_functor(transfer='reformed_spectra_densesapce_safe.hdf5', select=(12000, 18000), batch_size=1),
    steps=6000)

classifier.train(
    #input_fn=lambda: fn.input_functor(datanorm=testdatanorm, labelsnorm=testlabelsnorm, batch_size=1),
    input_fn=lambda: fn.input_hdf5_functor(transfer='reformed_spectra_densesapce_safe.hdf5', select=(18000, 24000), batch_size=1),
    steps=6000)

classifier.train(
    #input_fn=lambda: fn.input_functor(datanorm=testdatanorm, labelsnorm=testlabelsnorm, batch_size=1),
    input_fn=lambda: fn.input_hdf5_functor(transfer='reformed_spectra_densesapce_safe.hdf5', select=(24000, 29999), batch_size=1),
    steps=6000)
'''


def chunker(min_index=24000, max_index=100000, step=6000):
    for i in np.arange(min_index, max_index, step):
        yield (i, i + step)


#transfer = 'reformed_spectra_densesapce_safe.hdf5'
#transfer = 'reformed_TF_train_mp_1.hdf5'
transfer = 'reformed_TF_train_mp_1_quarter.hdf5'
step = 6000
min_index = 00000
max_index = 30000
chunks = chunker(min_index=min_index, max_index=max_index, step=step)
for chunk in chunks:
    classifier.train(
        input_fn=lambda: fn.input_hdf5_functor(transfer=transfer, select=chunk,
                                               batch_size=1),
        steps=step)
