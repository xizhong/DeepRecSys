# @Time  : 2022/4/7 21:55
# @Author: xizhong
# @Desc  :
import numpy as np
import pandas as pd


class Preprocessing():

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

    def replace_min_count(self, df, col_name, less_min_count_list, replace_str):
        # counter = df[col_name].value_counts(ascending=True)
        # less_min_count_list = []
        # for key, count in counter.iteritems():
        #     if count < min_count:
        #         less_min_count_list.append(key)
        #     else:
        #         break
        if less_min_count_list:
            if len(less_min_count_list) < df.shape[0]:
                values = [replace_str if _ in less_min_count_list else _ for _ in df[col_name].values]
                df[col_name] = pd.Series(values)
        return df[col_name]
