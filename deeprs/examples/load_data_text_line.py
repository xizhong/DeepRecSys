# @Time  : 2022/4/1 21:49
# @Author: xizhong
# @Desc  :


import tensorflow as tf

# tf.data.TextLineDataset -> Dataset
file_names = ['dataset_text_line.py']
batch_size = 128


def decode_line(line):
    features, label = str(line)[:-1], str(line)[-1]
    return features, label


def filter_line(line):
    if str(line) and len(str(line)) > 1:
        return True
    else:
        return False


dataset = tf.data.TextLineDataset(file_names, compression_type=None).filter(
    lambda line: filter_line(line)
).map(
    lambda line: decode_line(line)
).prefetch(batch_size).shuffle(buffer_size=batch_size)

iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)

# interleave 内部交织
