# @Time  : 2022/3/24 20:31
# @Author: xizhong
# @Desc  :

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from deeprs.layers.lr import LR

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

train_path = tf.keras.utils.get_file("criteo_sample.txt", '/tfr_data/home/zhongxi/PyWorkSpace/DeepRecSys/deeprs/tfr_data/criteo_sample.txt')

features = ['C' + str(i) for i in range(1, 27)] + ['I' + str(i) for i in range(1, 14)]
target = ['label']

def get_dataset(file_path):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=12,
      na_value="?",
      num_epochs=1,
      ignore_errors=True)
  return dataset


# train_x = tf.random.normal(shape=(100000, 10), dtype=tf.float32)
# weight = tf.reshape(tf.range(1, 11, 1, dtype=tf.float32), shape=(-1, 1))
# train_y = tf.matmul(train_x, weight) + 3
#
# tensors = train_x, train_y
# datasets = tf.tfr_data.Dataset.from_tensor_slices(tensors).shuffle(1000).batch(128)
#
# inputs = Input(shape=(10,), dtype=tf.float32)
#
# outputs = LR()(inputs)
#
# models = Model(inputs=inputs, outputs=outputs)
#
# models.compile(
#     optimizer='sgd',
#     losses=tf.keras.losses.MeanSquaredError(),
#     metrics=['mse'])
#
# models.fit(datasets, epochs=10, verbose=2)
