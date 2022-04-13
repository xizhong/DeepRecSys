# @Time  : 2022/3/28 20:54
# @Author: xizhong
# @Desc  :

import tensorflow as tf
from layers import LRLayer, EmbeddingLayer
from models import Model
from datasets import generate_feature_inputs_dict


def LR(feature_cols, params):
    inputs_dict = generate_feature_inputs_dict(feature_cols)
    inputs = list(inputs_dict.values())
    embed_dict = EmbeddingLayer(inputs_dict, params['embed_dim'])
    embed = tf.keras.layers.concatenate(embed_dict.values(), axis=1)
    outputs = LRLayer()(embed)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizer.get(params['optimizer'])(lr=params['lr'])
    model.compile(optimizer=optimizer, loss=params['losses'], metrics=params['metrics'])
    return model