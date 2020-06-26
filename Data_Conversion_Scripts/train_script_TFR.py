import numpy as np
import time

#base_filename = 'reformed_TF_train_mp_1_quarter'
#base_filename = 'TF_train_update_TFR'

#base_filename = 'TF_train_waveform_TFR'
#base_filename = 'TF_train_wave_unwrapped_TFR'
base_filename = 'TF_train_wave_unwrapped_eggs_TFR'
dataset_size = 60000
TFR_filesize = 10000

def file_chunker(start, stop, step, base_filename):
    for ii in np.arange(start, stop, step):
        yield '{}_{}-{}'.format(base_filename, ii, ii + step)


file_chunks = file_chunker(start=0, stop=dataset_size, step=TFR_filesize, base_filename=base_filename)

file_list = []
for file in file_chunks:
    file_list.append(file)

print(file_list)
repeat = 2
batch_size = 64
train_step = dataset_size*repeat

checkpoint = time.perf_counter()

classifier.train(
    input_fn=lambda: fn.input_TFR_functor(TFRecords_file_list=file_list, long=TFR_filesize, repeat=repeat, batch_size=batch_size),
    steps=train_step)

delta_t = checkpoint - time.perf_counter()
print('Trained {} epochs in {}'.format(repeat, delta_t))
