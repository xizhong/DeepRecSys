# @Time  : 2022/4/27 11:59
# @Author: xizhong
# @Desc  :
import collections
import re
import numpy as np


def convert_to_bucket(value):
    value = int(value)
    if value > 2:
        value = int(np.floor(np.log(value) ** 2))
    else:
        value = int(value)
    return str(value)


def load_data_dict(file_less_pth='../config/datasets/criteo_train_less_10.py',
              file_geq_pth='../config/datasets/criteo_train_dict_geq_10.py'):
    min_count_dict, uniq_vocab_dict = collections.OrderedDict(), collections.OrderedDict()
    pattern = r'[ \n\[\]\]"]'
    with open(file_less_pth) as f:
        for line in f.readlines():
            line = re.sub(pattern, '', line).replace(r',\n', '')
            name, seq = line.split(':')
            min_count_dict[name] = seq.split(',') if seq else list()
    with open(file_geq_pth) as f:
        for line in f.readlines():
            line = re.sub(pattern, '', line).replace(r',\n', '')
            name, seq = line.split(':')
            uniq_vocab_dict[name] = seq.split(',') if seq else list()
    for k, v in min_count_dict.items():
        if k.startswith('I'):
            uniq_vocab_dict[k].extend(v)
            min_count_dict[k] = list()
        min_count_dict[k] = set(min_count_dict[k])
    for k, v in uniq_vocab_dict.items():
        if k.startswith('I'):
            vv = list(set(map(convert_to_bucket, v)))
            vv.sort(key=lambda x: int(x))
            uniq_vocab_dict[k] = vv
    return min_count_dict, uniq_vocab_dict

