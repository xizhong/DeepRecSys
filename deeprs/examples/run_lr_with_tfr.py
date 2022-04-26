# @Time  : 2022/4/17 15:46
# @Author: xizhong
# @Desc  :

# import os
# import sys
# sys.path.append(os.path.dirname(sys.path[0]))

import tensorflow as tf
from utils import load_config, set_gpu, set_logger
from datasets import load_train_features, generate_tfrecord_iter
from models import LR
import logging

# tf.compat.v1.enable_eager_execution()

set_gpu([0])

model = 'lr'
data = 'criteo_x4'
exp_name = f'{model} with {data}'
logger = set_logger(logging.INFO, log_file=f'../logs/{exp_name}.log')

model_params = load_config('../config/models/LR.yaml')

data_params, _ = load_train_features(
    '../data/criteo_all/feature_data/')

train_iter = generate_tfrecord_iter(
    data_params['train_trf_pth'],
    data_params['feature_cols'],
    data_params['label_col'],
    model_params['batch_size'])
valid_iter, test_iter = generate_tfrecord_iter([
    data_params['valid_trf_pth'], data_params['test_trf_pth']],
    data_params['feature_cols'], data_params['label_col'],
    model_params['batch_size'], is_train_data=False)

model = LR(data_params['feature_cols'], model_params)

callbacks_mck = tf.keras.callbacks.ModelCheckpoint(**model_params['checkpoint'])
callbacks_tb = tf.keras.callbacks.TensorBoard(**model_params['tensorboard'])
callbacks_es = tf.keras.callbacks.EarlyStopping(**model_params['early_stopping'])

model.fit(train_iter, epochs=model_params['epochs'],
          verbose=model_params['verbose'], callbacks=[callbacks_mck, callbacks_tb, callbacks_es],
          validation_data=valid_iter)
del model
new_model = tf.keras.models.load_model(model_params['checkpoint']['filepath'])
new_model.evaluate(test_iter, verbose=model_params['verbose'])
