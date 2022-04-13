# @Time  : 2022/4/9 00:28
# @Author: xizhong
# @Desc  :

import tensorflow as tf


class Encoder(object):
    def __init__(self, feature_cols_name, uniq_vocabulary, oov_token='__OOV__', mask_token=None):
        self.name = feature_cols_name
        self.vocabulary = uniq_vocabulary
        self.string_lookup = self._fit()
        self.oov_token = oov_token
        self.mask_token = mask_token

    def _fit(self):
        string_lookup = tf.keras.layers.StringLookup(
            vocabulary=self.vocabulary,
            oov_token=self.oov_token, mask_token=self.mask_token)
        self.vocab_size = string_lookup.vocab_size()
        return string_lookup

    def transform(self, X):
        return self.string_lookup(X).numpy()

    def set_vocab(self, vocabulary):
        self.string_lookup.set_vocab(vocabulary)

    def save_vocab(self):
        with open(f'{self.name}.txt') as f:
            f.write(self.string_lookup.get_vocab())
