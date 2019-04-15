import importlib
import time
import Transform2TFRecord as T2TFR

importlib.reload(T2TFR)

# T2TFR.reformat_2_TFRecord_pulse(transfer='TF_train_waveform_convert.hdf5', TFRecord='TF_train_waveform_TFR',
#                          events_per_file=10000)

T2TFR.reformat_2_TFRecord_pulse(transfer='TF_train_wave_unwrapped_eggs.hdf5',
                                TFRecord='TF_train_wave_unwrapped_eggs_TFR',
                                events_per_file=10000)
