# @Time  : 2022/4/28 14:54
# @Author: xizhong
# @Desc  :

import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from datasets import build_tfrecord_dataset
from utils import load_config, set_logger
import time
import logging

# data = 'criteo_x4_001'
# data = 'criteo_x4_002'
data = 'criteo_tiny'

logger = set_logger(logging.INFO,
                    f'../data/{data}/gen_tfr_{data}_{time.strftime("%Y%m%d%H", time.localtime())}.log')
params = load_config(f'../datasets/config/{data.split("_")[0]}.yaml', data_name=data)

feature_cols, label_col = build_tfrecord_dataset(
    params['feature_cols'], params['label_col'], params['data_root'], params['tfr_data'],
    params['train_data'], params['output_feature'], params['tfr_data_size'], params['tfr_processes'],
    params['valid_data'], params['test_data'], params['file_less_pth'], params['file_geq_pth'], logger)




