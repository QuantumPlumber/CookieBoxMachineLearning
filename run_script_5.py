import tensorflow as tf
import numpy as np
import Functions as fn
import importlib

importlib.reload(fn)

# testdatanorm = np.random.rand(100, 16, 100)
# testlabelsnorm = np.random.rand(100, 100)

checkpointing_config = tf.estimator.RunConfig(
    save_checkpoints_secs=20 * 60,  # Save checkpoints every 20 minutes.
    keep_checkpoint_max=2,  # Retain the 10 most recent checkpoints.
    save_summary_steps=1000,
    log_step_count_steps=1000
)

classifier = tf.estimator.Estimator(

    model_fn=fn.CNNmodel,
    # model_dir='Model_large_sym',
    model_dir='scaled_sym_model',
    config=checkpointing_config,
    params={

        # 'feature_columns': the_feature_column,

        # Layers.
        'NUM_COOKIES': 16,
        'CNN': [[12, 10]],  # Convolutional layers
        'POOL': [10, 5],  # Global Pooling Label

        'DENSE': [2304, 1152, 1152, 500, 500, 200],  # Dense layers
        'OUT': 200  # output dimensions
    }
)
