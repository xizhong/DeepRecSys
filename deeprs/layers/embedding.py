# @Time  : 2022/4/12 11:42
# @Author: xizhong
# @Desc  :
import collections

import tensorflow as tf
from layers import Layer
from datasets import CategoricalCol, NumericCol, SequentialCol


class EmbeddingLayer(Layer):
    def __init__(self, feature_cols, embed_dim, **kwargs):
        super(EmbeddingLayer, self).__init__(**kwargs)
        self.embedding_layer = EmbeddingDictLayer(feature_cols, embed_dim)

    def call(self, inputs):
        return self.embedding_layer.call(inputs)


class EmbeddingDictLayer(Layer):
    def __init__(self, feature_cols, embed_dim, **kwargs):
        super(EmbeddingDictLayer, self).__init__(**kwargs)
        self.feature_cols = feature_cols
        self.embed_dim = embed_dim
        self.embedding_dict = self._generate_embedding_dict()

    def call(self, inputs):
        embed_dict = collections.OrderedDict()
        for name, feat in inputs.items():
            embed_dict[name] = self.embedding_dict[name](feat)
        return embed_dict

    def _generate_embedding_dict(self):
        embedding_dict = collections.OrderedDict()
        for _ in self.feature_cols:
            for feat in _:
                if isinstance(feat, CategoricalCol):
                    embedding = tf.keras.layers.Embedding(feat.vocab_size, self.embed_dim)
                elif isinstance(feat, NumericCol):
                    embedding = tf.keras.layers.Linear(1, self.embed_dim)
                elif isinstance(feat, SequentialCol):
                    embedding = embedding_dict[feat.share]
                else:
                    raise NotImplementedError("Type must be in [Categorical, Numeric, Sequential]")
                embedding_dict[feat.name] = embedding
        return embedding_dict


