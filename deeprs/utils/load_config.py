# @Time  : 2022/4/7 20:23
# @Author: xizhong
# @Desc  :

import yaml
import tensorflow as tf
import json
import os


def load_config(config_pth, data_name):
    params = dict()
    if config_pth is None or not config_pth:
        raise ValueError(
            "Make sure config_pth in [model_file_pth, data_file_pth]")
    assert os.path.isfile(config_pth) == 1, f'Invalid config file {config_pth}'
    with open(config_pth, 'r', encoding='utf-8') as f:
        params.update(yaml.load(f, Loader=yaml.FullLoader))
    return params[data_name]


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.keras.utils.set_random_seed(seed)


def set_gpu(gpus=None):
    if not gpus:
        tf.config.experimental.set_visible_devices(
            devices=[], device_type='GPU')
        return
    r_gpus = tf.config.experimental.list_physical_devices('GPU')
    if r_gpus:
        devices = [r_gpus[_] for _ in gpus if _ < len(r_gpus)]
        tf.config.experimental.set_visible_devices(
            devices=devices, device_type='GPU')
        # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        #     print(
        #         len(gpus),
        #         "Physical GPUs,",
        #         len(logical_gpus),
        #         "Logical GPU")


def dict_to_json(data):
    new_data = dict((k, str(v)) for k, v in data.items())
    return json.dumps(new_data, indent=4)
