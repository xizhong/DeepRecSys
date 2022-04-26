# @Time  : 2022/4/7 20:16
# @Author: xizhong
# @Desc  :
import logging
import os

# import sys
# sys.path.append(os.path.dirname(sys.path[0]))

import tensorflow as tf
from utils import load_config, set_gpu, set_logger
from datasets import build_tfrecord_dataset, generate_tfrecord_iter
from models import LR

set_gpu([1])

model = 'lr'
data = 'criteo_x4'
exp_name = f'{model}_{data}'
logger = set_logger(logging.INFO, f'../logs/{exp_name}.log')


params = load_config('../config/datasets/Criteo.yaml', '../config/models/LR.yaml')

feature_cols, label_col = build_tfrecord_dataset(
    params['feature_cols'], params['label_col'], params['data_root'], params['tfr_data'],
    params['train_data'], params['output_feature'], params['tfr_data_size'], params['valid_data'], params['test_data'],
    logger)

train_iter = generate_tfrecord_iter(
    os.path.join(params['data_root'], params['tfr_data'],
                 os.path.basename(params['train_data']).replace('csv', 'tfrd.*')),
    feature_cols, label_col, params['batch_size'])
valid_iter, test_iter = generate_tfrecord_iter([
    os.path.join(params['data_root'], params['tfr_data'],
                 os.path.basename(params['valid_data']).replace('csv', 'tfrd.*')),
    os.path.join(params['data_root'], params['tfr_data'],
                 os.path.basename(params['test_data']).replace('csv', 'tfrd.*'))
], feature_cols, label_col, params['batch_size'], is_train_data=False)

model = LR(feature_cols, params)

callbacks_mck = tf.keras.callbacks.ModelCheckpoint(**params['checkpoint'])
callbacks_tb = tf.keras.callbacks.TensorBoard(**params['tensorboard'])
callbacks_es = tf.keras.callbacks.EarlyStopping(**params['early_stopping'])

model.fit(train_iter, epochs=params['epochs'],
          verbose=params['verbose'], callbacks=[callbacks_mck, callbacks_tb, callbacks_es],
          validation_data=valid_iter)
del model
new_model = tf.keras.models.load_model(params['checkpoint']['filepath'])
new_model.evaluate(test_iter, verbose=params['verbose'])
