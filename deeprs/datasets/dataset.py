# @Time  : 2022/4/9 19:21
# @Author: xizhong
# @Desc  :

import collections
import logging
import os.path
import pandas as pd
from datasets import Preprocessing
from datasets import Encoder
from datasets import write_tfrecord, read_tfrecord
from datasets import generate_feature_cols

FeatureConfig = collections.namedtuple(
    'FeatureConfig', ['name', 'dtype', 'type', 'preprocess',
                      'na_value', 'min_count', 'max_len', 'share', 'source', 'vocab_size'])


def gen_feature_configs(feature_config_dict):
    feature_configs = list()
    for fcg in feature_config_dict:
        names = fcg['name']
        for name in names:
            fc = FeatureConfig(name,
                                        fcg.get('dtype', None),
                                        fcg.get('type', None),
                                        fcg.get('preprocess', None),
                                        fcg.get('na_value', None),
                                        fcg.get('min_count', None),
                                        fcg.get('max_len', None),
                                        fcg.get('share', None),
                                        fcg.get('source', None),
                                        None)
            feature_configs.append(fc)
    return feature_configs


def processing_df(df, feature_configs,
                  train_data=True, encoders=None, oov_token='__OOV__'):
    preprocessing = Preprocessing()
    if not encoders:
        encoders = dict()
    for feature_config in feature_configs:
        name = feature_config.name
        if feature_config.min_count is not None:
            df[name] = preprocessing.replace_min_count(df, name, feature_config.min_count, oov_token)
        if feature_config.na_value is not None:
            df[name] = preprocessing.fill_na(df, name, feature_config.na_value)
        if feature_config.preprocess is not None:
            pp_fn = getattr(preprocessing, feature_config.preprocess)
            df[name] = pp_fn(df, name)
        if feature_config.type == 'categorical':
            if train_data:
                uniq_vocab = df[name].unique()
                encoder = Encoder(name, uniq_vocab, oov_token)
                df[name] = encoder.transform(df[name])
                feature_config.vocab_size = len(uniq_vocab)
                encoder[name] = encoder
            else:
                df[name] = encoders[name].transform(df[name])
        elif feature_config.type == 'sequential':
            share_name = feature_config.categorical
            df[name] = encoders[share_name].transform(df[name])
    return df, encoders


def build_tfrecord_dataset(feature_cols_dict, label_col_dict,
                           train_data, valid_data=None, test_data=None):
    logging.info("Start build TFRecord dataset...")
    feature_configs = gen_feature_configs(feature_cols=feature_cols_dict)
    train_df = pd.read_csv(train_data)
    train_df, encoders = processing_df(train_df, feature_configs)
    feature_cols, label_col = generate_feature_cols(feature_configs, label_col_dict)
    write_tfrecord(f'{os.path.dirname(train_data)}/train.tfrecord', train_df, feature_cols, label_col)
    if valid_data:
        valid_df = pd.read_csv(valid_data)
        valid_df, _ = processing_df(valid_df, feature_configs, train_data=False, encoders=encoders)
        write_tfrecord(f'{os.path.dirname(train_data)}/valid.tfrecord', valid_df, label_col)
    if test_data:
        test_df = pd.read_csv(test_data)
        test_df, _ = processing_df(test_df, feature_configs, train_data=False, encoders=encoders)
        write_tfrecord(f'{os.path.dirname(train_data)}/test.tfrecord', test_df, label_col)
    return


def generate_tfrecord_iter(tfrecord_files, feature_cols, label_col, train_data=True, shuffle_factor=10):
    if not train_data:
        shuffle_factor = 0
    data_iter = read_tfrecord(tfrecord_files, feature_cols, label_col, shuffle_factor=shuffle_factor)
    return data_iter
