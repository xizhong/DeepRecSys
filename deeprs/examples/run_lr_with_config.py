# @Time  : 2022/4/7 20:16
# @Author: xizhong
# @Desc  :

from utils import load_config, set_gpu
from datasets import build_tfrecord_dataset, generate_tfrecord_iter
from models import LR

set_gpu([0])

model = 'lr'
data = 'criteo_x4'
exp_name = f'{model} with {data}'

params = load_config('../config/datasets/Criteo.yaml', '../config/models/LR.yaml')

# feature_cols, label_col = load_feature_cols(file_path)

feature_cols, label_col = build_tfrecord_dataset(
    params['feature_cols'], params['label_col'],
    params['train_data'], params['valid_data'], params['test_data']
)

train_iter = generate_tfrecord_iter("", feature_cols, label_col)
valid_iter, test_iter = generate_tfrecord_iter("", feature_cols, label_col, train_data=False)
model = LR(feature_cols, params)

model.fit(train_iter, epochs=1, verbose=params['verbose'], callbacks=None, validation_data=valid_iter)
model.predict(test_iter)






