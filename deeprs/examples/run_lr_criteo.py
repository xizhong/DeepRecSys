# @Time  : 2022/3/28 20:55
# @Author: xizhong
# @Desc  :

import pandas as pd
import tensorflow as tf
import time
from models.lr import LR
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

# tfr_data info
train_data_url = "../tfr_data/criteo_mini/raw_data/train.tiny.csv"
test_data_url = "../tfr_data/criteo_mini/raw_data/test.tiny.csv"
valid_data_url = "../tfr_data/criteo_mini/raw_data/valid.tiny.csv"

sparse_feature = ['C' + str(_) for _ in range(1, 27)]
dense_feature = ['I' + str(_) for _ in range(1, 14)]

sparse_feat_columns = [
    tf.keras.layers.Input(
        shape=(
            1,
        ),
        dtype='string',
        name=feat) for feat in sparse_feature]
dense_feat_columns = [
    tf.keras.layers.Input(
        shape=(
            1,
        ),
        dtype='float32',
        name=feat) for feat in dense_feature]

start = time.time()

# generate dataset

def dense_map_fun(x):
    if x <= 2:
        x = 1
    else:
        x = np.floor(np.log2(x))
    return x

def df_to_dataset(
        data_url,
        sparse_feature,
        dense_feature,
        shuffle=False,
        batch_size=1000,
        sparse_column=False):
    df = pd.read_csv(
        data_url)
    df[sparse_feature] = df[sparse_feature].fillna('OOV').astype(str)
    for feat in dense_feature:
        df[feat] = df[feat].astype(float).map(dense_map_fun)
    sparse_vocab_dict = dict()
    if sparse_column:
        for feat in sparse_feature:
            sparse_vocab_dict[feat] = df[feat].unique()
    target = df.pop('Label')
    ds = tf.data.Dataset.from_tensor_slices((dict(df), target))
    if shuffle:
        ds = ds.shuffle(buffer_size=5 * batch_size)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds, sparse_vocab_dict




train_dataset, sparse_column_dict = df_to_dataset(
    train_data_url, sparse_feature, dense_feature, shuffle=True, sparse_column=True)
test_dataset, _ = df_to_dataset(test_data_url, sparse_feature, dense_feature)
valid_dataset, _ = df_to_dataset(valid_data_url, sparse_feature, dense_feature)
print(f'Gen dataset Time cost {time.time() - start:.2f}')

model = LR(sparse_feat_columns, dense_feat_columns, sparse_column_dict)

model.compile('adam', tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True), metrics=['accuracy'])
model.fit(train_dataset, epochs=3)

result = model.evaluate(valid_dataset)
model.evaluate(test_dataset)
