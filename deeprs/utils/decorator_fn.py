# @Time  : 2022/4/1 23:05
# @Author: xizhong
# @Desc  :

import time


def execution_time(func):
    def inner(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        print(f'{str(func.__name__)} cost time: {end-start: .4f}s')
        return ret
    return inner
