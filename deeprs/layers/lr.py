# @Time  : 2022/3/24 20:12
# @Author: xizhong
# @Desc  :

import tensorflow as tf
from layers import Layer, EmbeddingLayer


class LRLayer(Layer):
    def __init__(self, feature_cols, embed_dim, regularizer='', activation: bool = False,  **kwargs):
        super(LRLayer, self).__init__(**kwargs)
        self.activation = activation
        self.feature_cols = feature_cols
        self.embed_dim = embed_dim
        self.regularizer = regularizer
        self.embedding_layer = EmbeddingLayer(self.feature_cols, self.embed_dim, self.regularizer)

    def build(self, input_shape):
        self.bias = self.add_weight('bias', shape=(1,), dtype=tf.float32, initializer='zero', trainable=True)
        super(LRLayer, self).build(input_shape)

    def call(self, inputs):
        embed_dict = self.embedding_layer(inputs)
        output = tf.keras.backend.sum(tf.keras.layers.Flatten()(
                tf.keras.layers.concatenate(list(embed_dict.values()), axis=1)),
            axis=1,
            keepdims=True) + self.bias
        if self.activation:
            output = tf.keras.activations.sigmoid(output)
        return output

    def compute_output_shape(self, input_shape):
        return (1,)

    def get_config(self):
        config = {'activation': self.activation, 'feature_cols': self.feature_cols, 'embed_dim': self.embed_dim,
                  'regularizer': self.regularizer}
        return config.update(super(LRLayer, self).get_config())

    @classmethod
    def from_config(cls, config):
        return cls(**config)
