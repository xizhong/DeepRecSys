# @Time  : 2022/4/17 15:46
# @Author: xizhong
# @Desc  :

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append(os.path.dirname(sys.path[0]))

from utils import load_config, set_gpu, set_logger, load_callbacks_fn
from datasets import load_train_features, generate_tfrecord_iter
from models import LR
import logging

# tf.compat.v1.enable_eager_execution()

model = 'lr'
data = 'criteo_x4_001'
# data = 'criteo_x4_002'
# data = 'criteo_tiny'
logger = set_logger(logging.INFO, log_file=f'../logs/{model}/{data}/{model}_with_{data}.log')
model_params = load_config(f'../config/{model}.yaml', f'{model}_{data}')

set_gpu(model_params['gpu'])

data_params, _ = load_train_features(
    f'../data/{data}/feature_data/')

train_iter = generate_tfrecord_iter(
    data_params['train_tfr_pth'],
    data_params['feature_cols'],
    data_params['label_col'],
    model_params['batch_size'])
valid_iter, test_iter = generate_tfrecord_iter([
    data_params['valid_tfr_pth'], data_params['test_tfr_pth']],
    data_params['feature_cols'], data_params['label_col'],
    model_params['batch_size'], is_train_data=False)


model = LR(data_params['feature_cols'], model_params)

callbacks = load_callbacks_fn(model_params['checkpoint'], model_params['tensorboard'], model_params['early_stopping'])

model.fit(train_iter, epochs=model_params['epochs'],
          verbose=model_params['verbose'], callbacks=callbacks, validation_data=valid_iter)
del model
new_model = LR(data_params['feature_cols'], model_params)
new_model.load_weights(model_params['checkpoint']['filepath'])
new_model.evaluate(test_iter, verbose=model_params['verbose'])
