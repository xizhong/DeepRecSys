# @Time  : 2022/3/15 20:02
# @Author: xizhong
# @Desc  :

import tensorflow as tf
from tensorflow.keras.layers import Layer


class FM(Layer):

    def __init__(self,
                 units,
                 **kwargs):
        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs):
        pass

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                'The last dimension of the input shape of a Dense layer '
                'should be defined. Found None. '
                f'Received: input_shape={input_shape}')
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        config = super(FM, self).get_config()
        config.update({
            'units': self.units
        })
        return config
