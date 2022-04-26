# @Time  : 2022/1/18 11:19 上午
# @Author: xizhong
# @Desc  :
"""
哈哈哈
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 生成虚拟数据
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))
data_iter = tf.data.Dataset.from_tensor_slices((data, labels)).batch(128)


# 训练模型，以 32 个样本为一个 batch 进行迭代
# model.fit(data, labels, epochs=10, batch_size=100)
model.fit(data_iter, epochs=10)
