# @Time  : 2022/3/15 20:02
# @Author: xizhong
# @Desc  :

import tensorflow as tf
from layers import Layer, LRLayer, EmbeddingLayer


class FMLayer(Layer):
    def __init__(self, feature_cols, embed_dim, regularizer='', activation: bool = False,  **kwargs):
        super(FMLayer, self).__init__(**kwargs)
        self.activation = activation
        self.feature_cols = feature_cols
        self.embed_dim = embed_dim
        self.regularizer = regularizer
        self.linear_logit = LRLayer(self.feature_cols, 1, '', False)
        self.embedding_layer = EmbeddingLayer(self.feature_cols, self.embed_dim, self.regularizer)

    def build(self, input_shape):
        super(FMLayer, self).build(input_shape)

    def call(self, inputs):
        linear_output = self.linear_logit(inputs)
        embed = (tf.keras.layers.concatenate(list(self.embedding_layer(inputs).values()), axis=1))
        interaction_output = 0.5 * tf.keras.backend.sum(
            tf.keras.backend.sum(embed, axis=1) ** 2 - tf.keras.backend.sum(embed ** 2, axis=1))
        output = linear_output + interaction_output
        if self.activation:
            output = tf.keras.activations.sigmoid(output)
        return output

    def compute_output_shape(self, input_shape):
        return (1, )

    def get_config(self):
        config = {'activation': self.activation, 'feature_cols': self.feature_cols, 'embed_dim': self.embed_dim,
                  'regularizer': self.regularizer}
        return config.update(super(LRLayer, self).get_config())
