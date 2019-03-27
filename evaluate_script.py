import numpy as np
import Functions as fn
import matplotlib.pyplot as plt
import h5py
import importlib

importlib.reload(fn)

# transfer='reformed_spectra_safe.hdf5'
# transfer = 'reformed_TF_train_mp_1.hdf5'
transfer = 'reformed_TF_train_widegate.hdf5'

for i in range(1):
    evaluations = classifier.evaluate(
        input_fn=lambda: fn.evaluate_hdf5_functor(transfer=transfer, select=(6500, 6501), batch_size=1))

    print('Current Evaluation loss is {}'.format(evaluations['loss']))

