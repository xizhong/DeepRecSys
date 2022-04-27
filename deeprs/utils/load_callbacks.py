# @Time  : 2022/4/26 12:27
# @Author: xizhong
# @Desc  :

import tensorflow as tf


def load_callbacks_fn(checkpoint=None, tensorboard=None, early_stopping=None):
    callbacks = list()
    if checkpoint is not None:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(**checkpoint))
    if tensorboard is not None:
        callbacks.append(tf.keras.callbacks.TensorBoard(**tensorboard))
    if early_stopping is not None:
        callbacks.append(tf.keras.callbacks.EarlyStopping(**early_stopping))
    return callbacks if len(callbacks) > 0 else None


