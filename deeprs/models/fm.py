# @Time  : 2022/3/24 20:09
# @Author: xizhong
# @Desc  :

from models import Model


class FM(Model):
    def __init__(self, dim=10, **kwargs):
        super(FM, self).__init__(**kwargs)
        self.dim = dim

    def call(self, inputs, training=None, mask=None):
        pass
