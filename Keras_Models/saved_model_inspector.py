import numpy as np
import h5py

direct = './multilayer_fc_truth'
filename = direct + '/' + 'saved_model.h5'

model_file = h5py.File(filename, 'r')

for key in model_file.keys():
    print(key)
    for keykey in model_file[key].keys():
        print(keykey)


model_file.close()