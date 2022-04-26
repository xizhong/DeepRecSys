# @Time  : 2022/4/9 19:21
# @Author: xizhong
# @Desc  :
import os
import pandas as pd
from datasets import Preprocessing
from datasets import Encoder, FeatureConfig
from datasets import write_tfrecord, read_tfrecord
from datasets import generate_feature_cols, save_train_features


def generate_feature_configs(feature_config_dict):
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
                  is_train_data=True, encoders=None, logger=None):
    preprocessing = Preprocessing()
    if not encoders:
        encoders = dict()
    for idx, feature_config in enumerate(feature_configs):
        logger.info(f'Preprocess feature: {feature_config.name}')
        name = feature_config.name
        if is_train_data and feature_config.min_count is not None:
            logger.debug(f'Preprocess feature: {feature_config.name} replace min count ...')
            df[name] = preprocessing.replace_min_count(df, name, feature_config.min_count, feature_config.na_value)
        if feature_config.na_value is not None:
            logger.debug(f'Preprocess feature: {feature_config.name} fill na ...')
            df[name] = preprocessing.fill_na(df, name, feature_config.na_value)
        if feature_config.preprocess is not None:
            logger.debug(f'Preprocess feature: {feature_config.name} {feature_config.preprocess} ...')
            pp_fn = getattr(preprocessing, feature_config.preprocess)
            df[name] = pp_fn(df, name)
        logger.debug(f'Preprocess feature: {feature_config.name} encode {feature_config.type} ...')
        if feature_config.type == 'categorical':
            if feature_config.dtype not in ['str', 'string']:
                df[name] = df[name].astype('int32', errors='ignore').astype(str)
            if is_train_data:
                uniq_vocab = df[name].unique()
                encoder = Encoder(name, uniq_vocab)
                df[name] = encoder.transform(df[name])
                feature_config = feature_config._replace(vocab_size=len(uniq_vocab) + 1)
                encoders[name] = encoder
                feature_configs[idx] = feature_config
            else:
                df[name] = encoders[name].transform(df[name])
        elif feature_config.type == 'sequential':
            share_name = feature_config.categorical
            df[name] = encoders[share_name].transform(df[name])
        else:
            raise NotImplementedError(f'Type: {feature_config.type} is not implemented')
        logger.info(f'Preprocess feature: {feature_config.name} done')
    return df, encoders


def build_tfrecord_dataset(feature_cols_dict, label_col_dict, data_root, tfr_data,
                           train_data, output_feature, tfr_data_size, valid_data=None, test_data=None, logger=None):
    logger.info("Start build TFRecord dataset...")
    feature_configs = generate_feature_configs(feature_config_dict=feature_cols_dict)
    train_df = pd.read_csv(train_data)
    logger.info("Build train tfr_data TFRecord dataset: preprocess...")
    train_df, encoders = processing_df(train_df, feature_configs)
    feature_cols, label_col = generate_feature_cols(feature_configs, label_col_dict)
    train_trf_pth, valid_trf_pth, test_trf_pth = get_tfrecord_path(data_root, tfr_data, train_data, valid_data, test_data)
    logger.info("Build train tfr_data TFRecord dataset: save...")
    save_train_features(feature_cols, label_col, os.path.join(data_root, output_feature), encoders,
                        train_trf_pth, valid_trf_pth, test_trf_pth)
    logger.info("Build train tfr_data TFRecord dataset: writing...")
    write_tfrecord(train_trf_pth, train_df, feature_cols, label_col, tfr_data_size)
    logger.info("Build train tfr_data TFRecord dataset: done")
    if valid_data:
        valid_df = pd.read_csv(valid_data)
        logger.info("Build valid tfr_data TFRecord dataset: preprocess...")
        valid_df, _ = processing_df(valid_df, feature_configs, is_train_data=False, encoders=encoders)
        logger.info("Build valid tfr_data TFRecord dataset: writing")
        write_tfrecord(valid_trf_pth, valid_df, feature_cols, label_col, tfr_data_size)
        logger.info("Build valid tfr_data TFRecord dataset: done")
    if test_data:
        test_df = pd.read_csv(test_data)
        logger.info("Build test tfr_data TFRecord dataset: preprocess...")
        test_df, _ = processing_df(test_df, feature_configs, is_train_data=False, encoders=encoders)
        logger.info("Build test tfr_data TFRecord dataset: writing")
        write_tfrecord(test_trf_pth, test_df, feature_cols, label_col, tfr_data_size)
        logger.info("Build test tfr_data TFRecord dataset: done")
    return feature_cols, label_col


def generate_tfrecord_iter(tfrecord_files, feature_cols, label_col, batch_size, is_train_data=True, shuffle_factor=10):
    if not is_train_data:
        shuffle_factor = 0
    if isinstance(tfrecord_files, str):
        return read_tfrecord(tfrecord_files, feature_cols, label_col, batch_size=batch_size, shuffle_factor=shuffle_factor)
    else:
        iter_list = list()
        for tfr in tfrecord_files:
            iter_list.append(
                read_tfrecord(tfr, feature_cols, label_col, batch_size=batch_size, shuffle_factor=shuffle_factor))
    return iter_list


def get_tfrecord_path(data_root, tfr_data, train_data, valid_data, test_data):
    ret = list()
    for pth in [train_data, valid_data, test_data]:
        if pth is not None:
            ret.append(os.path.join(data_root, tfr_data, os.path.basename(pth).replace('csv', 'tfrd')))
        else:
            ret.append(None)
    return ret

