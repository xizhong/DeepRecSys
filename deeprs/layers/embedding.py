# @Time  : 2022/4/12 11:42
# @Author: xizhong
# @Desc  :
import collections

import tensorflow as tf
from layers import Layer
from datasets import CategoricalCol, NumericCol, SequentialCol


class EmbeddingLayer(Layer):
    def __init__(self, inputs_dict, embed_dim, **kwargs):
        super(EmbeddingLayer, self).__init__(**kwargs)
        self.embedding_layer = EmbeddingDictLayer(inputs_dict, embed_dim)

    def call(self, inputs):
        return self.embedding_layer(inputs)


class EmbeddingDictLayer(Layer):
    def __init__(self, inputs_dict, embed_dim, **kwargs):
        super(EmbeddingDictLayer, self).__init__(**kwargs)
        self.inputs_dict = inputs_dict
        self.embed_dim = embed_dim
        self.embedding_dict = self._generate_embedding_dict()

    def call(self, inputs):
        embed_dict = collections.OrderedDict()
        for name, feature in inputs:
            embed_dict[name] = self.embedding_dict[name](feature)
        return embed_dict

    def _generate_embedding_dict(self):
        embedding_dict = collections.OrderedDict()
        for name, feat in self.inputs_dict:
            if isinstance(feat, CategoricalCol):
                embedding = tf.keras.Layers.Embedding(feat.vocab_size, self.embed_dim)
            elif isinstance(feat, NumericCol):
                embedding = tf.keras.Layers.Linear(1, self.embed_dim)
            elif isinstance(feat, SequentialCol):
                embedding = embedding_dict[feat.share]
            else:
                raise NotImplementedError("Type must be in [Categorical, Numeric, Sequential]")
            embedding_dict[name] = embedding
        return embedding_dict


