# @Time  : 2022/4/12 11:42
# @Author: xizhong
# @Desc  :
import collections

import tensorflow as tf
from layers import Layer
from datasets import CategoricalCol, NumericCol, SequentialCol


class EmbeddingLayer(Layer):
    def __init__(self, feature_cols, embed_dim, embeddings_regularizer, **kwargs):
        super(EmbeddingLayer, self).__init__(**kwargs)
        self.feature_cols = feature_cols
        self.embed_dim = embed_dim
        self.embedding_regularizer = embeddings_regularizer
        self.embedding_layer = EmbeddingDictLayer(feature_cols, embed_dim, embeddings_regularizer)

    def build(self, input_shape):
        super(EmbeddingLayer, self).build(input_shape)

    def call(self, inputs):
        return self.embedding_layer.call(inputs)

    def get_config(self):
        config = {'feature_cols': self.feature_cols, 'embed_dim': self.embed_dim,
                  'embedding_regularizer': self.embedding_regularizer}
        return config.update(super(EmbeddingLayer, self).get_config())

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class EmbeddingDictLayer(Layer):
    def __init__(self, feature_cols, embed_dim, embeddings_regularizer, **kwargs):
        super(EmbeddingDictLayer, self).__init__(**kwargs)
        self.feature_cols = feature_cols
        self.embed_dim = embed_dim
        self.embeddings_regularizer = embeddings_regularizer
        self.embedding_dict = self._generate_embedding_dict()

    def build(self, input_shape):
        super(EmbeddingDictLayer, self).build(input_shape)

    def call(self, inputs):
        embed_dict = collections.OrderedDict()
        for name, feat in inputs.items():
            embed_dict[name] = self.embedding_dict[name](feat)
        return embed_dict

    def _generate_embedding_dict(self):
        embeddings_regularizer = self.get_regularizer(self.embeddings_regularizer)
        embedding_dict = collections.OrderedDict()
        for _ in self.feature_cols:
            for feat in _:
                if isinstance(feat, CategoricalCol):
                    embedding = tf.keras.layers.Embedding(feat.vocab_size, self.embed_dim,
                                                          embeddings_regularizer=embeddings_regularizer)
                elif isinstance(feat, NumericCol):
                    embedding = tf.keras.layers.Linear(1, self.embed_dim)
                elif isinstance(feat, SequentialCol):
                    embedding = embedding_dict[feat.share]
                else:
                    raise NotImplementedError("Type must be in [Categorical, Numeric, Sequential]")
                embedding_dict[feat.name] = embedding
        return embedding_dict

    def get_config(self):
        config = {'feature_cols': self.feature_cols, 'embed_dim': self.embed_dim,
                  'embeddings_regularizer': self.embeddings_regularizer}
        return config.update(super(EmbeddingDictLayer, self).get_config())

    @classmethod
    def from_config(cls, config):
        return cls(**config)


