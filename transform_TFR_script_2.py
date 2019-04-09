import importlib
import time
import Transform2TFRecord as T2TFR

importlib.reload(T2TFR)

T2TFR.reformat_2_TFRecord_ri(transfer='TF_train_update.hdf5', TFRecord='TF_train_update_reformed',
                          events_per_file=10000)
