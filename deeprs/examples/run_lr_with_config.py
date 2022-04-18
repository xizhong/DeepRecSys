# @Time  : 2022/4/7 20:16
# @Author: xizhong
# @Desc  :
import os.path
import tensorflow as tf
from utils import load_config, set_gpu
from datasets import build_tfrecord_dataset, generate_tfrecord_iter
from models import LR
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')

set_gpu([0])

model = 'lr'
data = 'criteo_x4'
exp_name = f'{model} with {data}'

params = load_config('../config/datasets/Criteo.yaml', '../config/models/LR.yaml')

# feature_cols, label_col = load_feature_cols(file_path)

feature_cols, label_col = build_tfrecord_dataset(
    params['feature_cols'], params['label_col'], params['data_root'], params['tfr_data'],
    params['train_data'], params['valid_data'], params['test_data'])

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

model.fit(train_iter[0], train_iter[1], epochs=params['epochs'],
          verbose=params['verbose'], callbacks=None, validation_data=valid_iter)
model.evaluate(test_iter[0], test_iter[1])
