# @Time  : 2022/4/6 23:23
# @Author: xizhong
# @Desc  :

import tensorflow as tf


class Layer(tf.keras.layers.Layer):

    def get_regularizer(self, reg):
        if reg.startswith('l1_'):
            return tf.keras.regularizers.l1(float(reg.split('_')[1]))
        elif reg.startswith('l2_'):
            return tf.keras.regularizers.l2(float(reg.split('_')[1]))
        elif reg.startswith('l12_'):
            regs = reg.split('_')
            return tf.keras.regularizers.l1_l2(l1=float(regs[1]), l2=float(reg[1]))
