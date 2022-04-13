# @Time  : 2022/3/24 14:10
# @Author: xizhong
# @Desc  :

import collections
import pickle
from datasets import FeatureConfig
import tensorflow as tf

CategoricalCol = collections.namedtuple(
    'CategoricalCol', ['name', 'dtype', 'dim', 'vocab_size', 'source'])

NumericCol = collections.namedtuple(
    'NumericCol', ['name', 'dtype', 'source'])

SequentialCol = collections.namedtuple(
    'SequentialCol', ['name', 'max_len', 'share_col'])


def generate_feature_cols(feature_configs: list[FeatureConfig], label_col, file_path='feature.cols'):
    categorical_cols, numeric_cols, sequential_cols \
        = list[CategoricalCol], list[NumericCol], list[SequentialCol]
    for feature_config in feature_configs:
        if feature_config.type == 'categorical':
            col = CategoricalCol(
                feature_config.name,
                feature_config.dtype,
                None,
                feature_config.vocab_size,
                feature_config.source)
            categorical_cols.append(col)
        elif feature_config.type == 'numeric':
            col = NumericCol(
                feature_config.name,
                feature_config.dtype,
                feature_config.source)
            numeric_cols.append(col)
        elif feature_config.type == 'sequential':
            col = SequentialCol(
                feature_config.name,
                feature_config.max_len,
                feature_config.share)
            sequential_cols.append(col)
        else:
            raise NotImplementedError(
                "Type must be in [categorical, numeric, sequential]")
    with open(file_path, 'wb') as f:
        pickle.dump([categorical_cols, numeric_cols, sequential_cols, label_col], f)
    return [categorical_cols, numeric_cols, sequential_cols], label_col


def load_feature_cols(file):
    return pickle.load(file)


def generate_feature_inputs_dict(features_cols):
    inputs_dict = collections.OrderedDict()
    for feat in features_cols[0]:
        inputs_dict[feat['name']] = tf.keras.layers.Input(name=feat['name'], shape=(1, ), dtype=tf.int32)
    for feat in features_cols[1]:
        inputs_dict[feat['name']] = tf.keras.layers.Input(name=feat['name'], shape=(1, ), dtype=tf.float32)
    for feat in features_cols[2]:
        inputs_dict[feat['name']] = tf.keras.layers.Input(name=feat['name'], shape=(feat.max_len,), dtype=tf.int32)
    return inputs_dict

