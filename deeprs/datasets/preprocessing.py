# @Time  : 2022/4/7 21:55
# @Author: xizhong
# @Desc  :
import numpy as np
import logging


class Preprocessing(object):
    def __init__(self):
        pass

    def convert_to_bucket(self, df, col_name):
        def _convert_to_bucket(value):
            if value > 2:
                value = int(np.floor(np.log(value) ** 2))
            else:
                value = int(value)
            return value
        return df[col_name].map(_convert_to_bucket).astype(int)

    def fill_na(self, df, col_name, na_value):
        return df[col_name].fillna(na_value)

    def replace_min_count(self, df, col_name, min_count, replace_str):
        counter = df[col_name].value_counts(ascending=True)
        less_min_count_list = []
        for key, count in counter.iteritems():
            if count < min_count:
                less_min_count_list.append(key)
            else:
                break
        if less_min_count_list:
            df[col_name].replace(less_min_count_list, replace_str, inplace=True)
        return df[col_name]
