# @Time  : 2022/3/24 20:12
# @Author: xizhong
# @Desc  :

import tensorflow as tf
from layers import Layer


class LRLayer(Layer):
    def __init__(self, activation: bool = False, **kwargs):
        super(LRLayer, self).__init__(**kwargs)
        self.activation = activation

    def build(self, input_shape):
        self.weight = self.add_weight('weight',
                                      shape=(input_shape[-1], 1),
                                      dtype=tf.float32,
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.bias = self.add_weight(
            'bias',
            shape=(1,),
            dtype=tf.float32,
            initializer='zero',
            trainable=True
        )
        super(LRLayer, self).build(input_shape)

    def call(self, inputs):
        output = tf.matmul(inputs, self.weight) + self.bias
        if self.activation:
            output = tf.keras.activations.sigmoid(output)
        return output

    def compute_output_shape(self, input_shape):
        return self.bias.shape

    def get_config(self):
        config = super(LRLayer, self).get_config()
        return config.update({'activation': self.activation})
