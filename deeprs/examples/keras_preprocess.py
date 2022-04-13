# @Time  : 2022/3/25 21:48
# @Author: xizhong
# @Desc  :

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import time

start = time.time()
dataset_url = "/Users/zhongxi/Workspace/PycharmProject/DeepRecSys/deeprs/data/Criteo_x4/train.csv"
df = pd.read_csv(dataset_url)
print(f'Time cost {time.time() - start:.2f}')
sparse_feature = ['C' + str(_) for _ in range(1, 27)]
dense_feature = ['I' + str(_) for _ in range(1, 14)]

for feat in sparse_feature:
    df[feat] = df[feat].fillna('-1')
print(f'Time cost {time.time() - start:.2f}')
for feat in dense_feature:
    df[feat] = df[feat].fillna(-1)
print(f'Time cost {time.time() - start:.2f}')

train, test = train_test_split(df, test_size=0.2)
train, val = train_test_split(df, test_size=0.2)

print(df.head(10))
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

def df_to_dataset(df, shuffle=True, batch_size=128):
    df = df.copy()
    target = df.pop('Label')
    ds = tf.data.Dataset.from_tensor_slices((dict(df), target))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds

train_dataset = df_to_dataset(train, batch_size=128)