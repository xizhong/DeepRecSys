# @Time  : 2022/3/28 20:54
# @Author: xizhong
# @Desc  :

import tensorflow as tf
from layers import EmbeddingLayer
from models import Model
from datasets import generate_feature_inputs_dict


def LR(feature_cols, params):
    inputs_dict = generate_feature_inputs_dict(feature_cols)
    inputs = list(inputs_dict.values())
    embedding_layer = EmbeddingLayer(feature_cols, params['embed_dim'])
    embed_dict = embedding_layer(inputs_dict)
    embed = tf.keras.backend.sum(
        tf.keras.layers.Flatten()(
            tf.keras.layers.concatenate(list(embed_dict.values()),axis=1)),
        axis=1,
        keepdims=True)
    outputs = tf.keras.activations.sigmoid(embed)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.get(params['optimizer'])
    optimizer.lr = params['learning_rate']
    model.compile(
        optimizer=optimizer,
        loss=params['loss'],
        metrics=params['metrics'])
    return model
