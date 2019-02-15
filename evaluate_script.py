import numpy as np
import Functions as fn
import matplotlib.pyplot as plt
import h5py
import importlib

importlib.reload(fn)

for i in range(3):
    evaluations = classifier.evaluate(
        input_fn=lambda: fn.evaluate_hdf5_functor(transfer='reformed_spectra_safe.hdf5', select=(0, 100), batch_size=1))


