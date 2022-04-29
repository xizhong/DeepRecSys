# @Time  : 2022/3/24 14:10
# @Author: xizhong
# @Desc  :

import collections
import os.path
import pickle
import tensorflow as tf
from datasets import Encoder

FeatureConfig = collections.namedtuple(
    'FeatureConfig', ['name', 'dtype', 'type', 'preprocess',
                      'na_value', 'min_count', 'max_len', 'share', 'source', 'vocab_size'])

CategoricalCol = collections.namedtuple(
    'CategoricalCol', ['name', 'dtype', 'dim', 'vocab_size', 'source'])

NumericCol = collections.namedtuple(
    'NumericCol', ['name', 'dtype', 'source'])

SequentialCol = collections.namedtuple(
    'SequentialCol', ['name', 'max_len', 'share_col'])


def generate_feature_cols(feature_configs: list[FeatureConfig], label_col, file_path='feature.cols'):
    categorical_cols, numeric_cols, sequential_cols \
        = list[CategoricalCol](), list[NumericCol](), list[SequentialCol]()
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
    return [categorical_cols, numeric_cols, sequential_cols], label_col


def save_train_features(feature_cols, label_col, file_pth, encoders=None, train_tfr_pth=None, valid_tfr_pth=None, test_tfr_pth=None):
    if encoders is not None:
        encoder_dict = collections.OrderedDict()
        for name, encoder in encoders.items():
            encoder_dict[name] = encoder.get_vocab()
        with open(os.path.join(file_pth, 'encoders.cfg'), 'wb') as f:
            pickle.dump(encoder_dict, f)
    params = {
        'feature_cols': feature_cols,
        'label_col': label_col,
        'train_tfr_pth': train_tfr_pth + '.*',
        'valid_tfr_pth': valid_tfr_pth + '.*',
        'test_tfr_pth': test_tfr_pth + '.*'
    }
    with open(os.path.join(file_pth, 'train_features.cfg'), 'wb') as f:
        pickle.dump(params, f)


def load_train_features(file_pth, load_encoders=False):
    params, encoders = None, None
    train_features_pth = os.path.join(file_pth, 'train_features.cfg')
    if os.path.exists(train_features_pth):
        with open(train_features_pth, 'rb') as f:
            params = pickle.load(f, encoding='bytes')
    encoders_pth = os.path.join(file_pth, 'encoders.cfg')
    if load_encoders:
        if os.path.exists(encoders_pth):
            with open(encoders_pth, 'rb') as f:
                encoders = pickle.load(f, encoding='bytes')
            for name, vocab in encoders.items():
                encoder = Encoder(name, vocab)
                encoders[name] = encoder
    return params, encoders


def generate_feature_inputs_dict(feature_cols):
    inputs_dict = collections.OrderedDict()
    for feat in feature_cols[0]:
        inputs_dict[feat.name] = tf.keras.layers.Input(name=feat.name, shape=(1, ), dtype=tf.int32)
    for feat in feature_cols[1]:
        inputs_dict[feat.name] = tf.keras.layers.Input(name=feat.name, shape=(1, ), dtype=tf.float32)
    for feat in feature_cols[2]:
        inputs_dict[feat.name] = tf.keras.layers.Input(name=feat.name, shape=(feat.max_len,), dtype=tf.int32)
    return inputs_dict

