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


def chunker(step=5000, indexes=(0, 1), random_shuffled_index=(0, 1)):
    for i in indexes:
        yield np.sort(random_shuffled_index[i: i + step])


# transfer = 'reformed_spectra_densesapce_safe.hdf5'
# transfer = 'reformed_TF_train_mp_1.hdf5'
transfer = 'reformed_TF_train_mp_1_quarter.hdf5'
# transfer = 'reformed_TF_train_widegate.hdf5'

step = 100000
min_index = 00000
max_index = 100000
indexes = np.arange(min_index, max_index, step)
random_shuffled_index = np.arange(min_index, max_index)
np.random.shuffle(random_shuffled_index)
chunks = chunker(step=step, indexes=indexes, random_shuffled_index=random_shuffled_index)

batch_size = 64
train_step = step // batch_size
for chunk in chunks:
    classifier.train(
        input_fn=lambda: fn.input_hdf5_functor(transfer=transfer, select=chunk, batch_size=batch_size),
        steps=train_step)
