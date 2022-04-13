# @Time  : 2022/3/28 20:54
# @Author: xizhong
# @Desc  :

import tensorflow as tf
from layers import LRLayer
from models import Model
from datasets import generate_feature_inputs_dict


def LR(feature_cols, params):
    inputs_dict = generate_feature_inputs_dict(feature_cols)
    inputs = inputs_dict.values()
    embedding_layer = ss

    for feat, uniq_vocables in sparse_column_dict.items():
        embedding_lookup[feat] = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=uniq_vocables, mask_token=None),
            tf.keras.layers.Embedding(len(uniq_vocables), 32),
        ])
    sparse_layer_list = list()
    for feat in sparse_feat_columns:
        sparse_layer_list.append(tf.keras.backend.squeeze(embedding_lookup[feat.name](feat), axis=1))
    dense_list = list()
    for feat in dense_feat_columns:
        dense_list.append(feat)
    sparse_embedding_inputs = tf.keras.layers.concatenate(
        sparse_layer_list + dense_list, axis=1)
    outputs = LRLayer()(sparse_embedding_inputs)
    model = Model(inputs=inputs, outputs=outputs)
    return model

