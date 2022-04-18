# @Time  : 2022/4/17 15:46
# @Author: xizhong
# @Desc  :

import os.path
import tensorflow as tf
from utils import load_config, set_gpu
from datasets import load_train_features, generate_tfrecord_iter
from models import LR
import logging
tf.compat.v1.enable_eager_execution()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S')

set_gpu([0])

model = 'lr'
data = 'criteo_x4'
exp_name = f'{model} with {data}'

model_params = load_config('../config/models/LR.yaml')

data_params, _ = load_train_features(
    '../data/criteo_mini/tfr_data/')

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

model.fit(
    train_iter[0],
    train_iter[1],
    epochs=model_params['epochs'],
    verbose=model_params['verbose'],
    callbacks=None,
    validation_data=valid_iter)
model.evaluate(test_iter[0], test_iter[1])
