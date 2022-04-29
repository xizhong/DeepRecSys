# @Time  : 2022/4/9 19:21
# @Author: xizhong
# @Desc  :
import collections
import os
import pandas as pd
from datasets import Preprocessing
from datasets import Encoder, FeatureConfig
from datasets import write_tfrecord, read_tfrecord
from datasets import generate_feature_cols, save_train_features
from datasets import load_criteo_data_dict
import multiprocessing


def generate_feature_configs(feature_config_dict):
    feature_configs = list()
    for fcg in feature_config_dict:
        names = fcg['name']
        for name in names:
            fc = FeatureConfig(name, fcg.get('dtype', None), fcg.get('type', None), fcg.get('datasets', None),
                               fcg.get('na_value', None), fcg.get('min_count', None), fcg.get('max_len', None),
                               fcg.get('share', None), fcg.get('source', None), None)
            feature_configs.append(fc)
    return feature_configs


def processing_df(df, feature_configs, encoders=None, min_count_dict=None, logger=None):
    preprocessing = Preprocessing()
    for idx, feature_config in enumerate(feature_configs):
        logger.debug(f'Preprocess feature: {feature_config.name}')
        name = feature_config.name
        if feature_config.min_count is not None:
            logger.debug(f'Preprocess feature: {feature_config.name} replace min count ...')
            df[name] = preprocessing.replace_min_count(df, name, min_count_dict[name], feature_config.na_value)
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
            df[name] = encoders[name].transform(df[name])
        elif feature_config.type == 'sequential':
            share_name = feature_config.categorical
            df[name] = encoders[share_name].transform(df[name])
        else:
            raise NotImplementedError(f'Type: {feature_config.type} is not implemented')
        logger.debug(f'Preprocess feature: {feature_config.name} done')
    return df


def build_tfrecord_dataset(feature_cols_dict, label_col_dict, data_root, tfr_data, train_data, output_feature,
                           tfr_data_size, processes, valid_data=None, test_data=None,
                           file_less_pth=None, file_geq_pth=None, logger=None):
    logger.info("Start build TFRecord datasets...")
    feature_configs = generate_feature_configs(feature_config_dict=feature_cols_dict)
    logger.info("Loading data dict config...")
    min_count_dict, uniq_vocab_dict = load_criteo_data_dict(file_less_pth, file_geq_pth)
    logger.info("Generating categorical feature encoders...")
    encoders, feature_configs = get_encoder(uniq_vocab_dict, feature_configs)
    feature_cols, label_col = generate_feature_cols(feature_configs, label_col_dict)
    train_tfr_pth, valid_tfr_pth, test_tfr_pth = get_tfrecord_path(data_root, tfr_data,
                                                                   train_data, valid_data, test_data)
    logger.info("Saving features to file...")
    save_train_features(feature_cols, label_col, os.path.join(data_root, output_feature), encoders,
                        train_tfr_pth, valid_tfr_pth, test_tfr_pth)
    logger.info(feature_cols)
    writer_tfread_from_read_csv(train_data, 'train', feature_configs, train_tfr_pth, feature_cols, label_col,
                                tfr_data_size, encoders, min_count_dict, processes, logger)
    if valid_data:
        writer_tfread_from_read_csv(valid_data, 'valid', feature_configs, valid_tfr_pth, feature_cols, label_col,
                                    tfr_data_size, encoders, min_count_dict, processes, logger)
    if test_data:
        writer_tfread_from_read_csv(test_data, 'test', feature_configs, test_tfr_pth, feature_cols, label_col,
                                    tfr_data_size, encoders, min_count_dict, processes, logger)
    logger.info("End build TFRecord datasets.")
    return feature_cols, label_col


def generate_tfrecord_iter(tfrecord_files, feature_cols, label_col, batch_size, is_train_data=True, shuffle_factor=10):
    if not is_train_data:
        shuffle_factor = 0
    if isinstance(tfrecord_files, str):
        return read_tfrecord(tfrecord_files, feature_cols, label_col, batch_size=batch_size,
                             shuffle_factor=shuffle_factor)
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
            ret.append(os.path.join(data_root, tfr_data,
                    os.path.basename(pth).replace('csv', 'tfrd')))
        else:
            ret.append(None)
    return ret


def writer_tfread_from_read_csv(data, data_type, feature_configs, tfr_pth, feature_cols, label_col,
                                chunksize, encoders, min_count_dict, processes, logger):
    pool = multiprocessing.Pool(processes=processes)
    logger.info(f'Build {data_type} tfr_data TFRecord datasets: datasets...')
    idx = 1
    for df in pd.read_csv(data, chunksize=chunksize, iterator=True):
        logger.info(
            f'Build {data_type} {idx:02d} tfr_data TFRecord datasets: processing...')
        df = processing_df(df, feature_configs, encoders, min_count_dict, logger)
        pool.apply_async(
            single_process_func, (idx, data_type, df, feature_cols, label_col, tfr_pth, logger))
        idx += 1
    pool.close()
    pool.join()
    logger.info(f'Build {data_type} tfr_data TFRecord datasets: done')


def single_process_func(idx, data_type, df, feature_cols, label_col,
                        tfr_pth, logger):
    write_tfrecord(f'{tfr_pth}.{idx:02d}', df, feature_cols, label_col)
    logger.info(f'Build {data_type} {idx:02d} tfr_data TFRecord datasets: done')


def get_encoder(uniq_vocab_dict, feature_configs):
    encoders = collections.OrderedDict()
    for name, uniq_vocab in uniq_vocab_dict.items():
        encoder = Encoder(name, uniq_vocab, '__oov__')
        encoders[name] = encoder
    for idx, feature_config in enumerate(feature_configs):
        feature_config = feature_config._replace(
            vocab_size=encoders[feature_config.name].vocab_size)
        feature_configs[idx] = feature_config
    return encoders, feature_configs
