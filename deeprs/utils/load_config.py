# @Time  : 2022/4/7 20:23
# @Author: xizhong
# @Desc  :

import os
import yaml
import tensorflow as tf
import json


def load_config(*config_pth):
    params = dict()
    if not config_pth or len(config_pth) != 2:
        raise ValueError(
            "Make sure config_pth = [model_file_pth, data_file_pth]")
    for pth in config_pth:
        assert os.path.isfile(pth) == 1, f'Invalid config file {pth}'
        with open(pth, 'r', encoding='utf-8') as f:
            params.update(yaml.load(f, Loader=yaml.FullLoader))
    return params


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
