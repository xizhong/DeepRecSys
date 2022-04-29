# @Time  : 2022/3/28 20:54
# @Author: xizhong
# @Desc  :

import tensorflow as tf
from layers import LRLayer
from models import Model
from datasets import generate_feature_inputs_dict


def LR(feature_cols, params):
    inputs_dict = generate_feature_inputs_dict(feature_cols)
    inputs = list(inputs_dict.values())
    outputs = LRLayer(feature_cols, params['embed_dim'], params['regularizer'], True)(inputs_dict)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.get(params['optimizer'])
    optimizer.lr = params['learning_rate']
    model.compile(
        optimizer=optimizer,
        loss=params['loss'],
        metrics=params['metrics'])
    return model
