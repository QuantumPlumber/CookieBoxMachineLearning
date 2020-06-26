import importlib
import time
import Transform2TFRecord as T2TFR

importlib.reload(T2TFR)

T2TFR.reformat_2_TFRecord(transfer='reformed_TF_train_mp_1_quarter.hdf5', TFRecord='reformed_TF_train_mp_1_quarter',
                          events_per_file=10000)
