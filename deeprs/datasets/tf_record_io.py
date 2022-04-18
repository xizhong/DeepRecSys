# @Time  : 2022/3/25 12:02
# @Author: xizhong
# @Desc  :
import glob

from tqdm import tqdm
import tensorflow as tf
from utils import execution_time


@execution_time
def write_tfrecord(filename_pre, df, feature_cols, label_col, tfrecord_size=None):
    """
    write csv to tfrecord
    :param filename_pre: multi file pre name
    :param df: dataframe
    :param feature_cols:
    :param label_col:
    :param tfrecord_size: 1000 000
    :return:
    """
    def _make_example(line, feature_cols, label_col):
        features = {feat.name: tf.train.Feature(int64_list=tf.train.Int64List(
            value=[line[feat.name]])) for feat in feature_cols[0]}
        features.update({feat.name: tf.train.Feature(float_list=tf.train.FloatList(
                value=[line[feat.name]])) for feat in feature_cols[1]})
        features.update({feat.name: tf.train.Feature(int64_list=tf.train.Int64List(
                value=[line[feat.name]])) for feat in feature_cols[2]})
        features[label_col['name']] = tf.train.Feature(
            float_list=tf.train.FloatList(value=[line[label_col['name']]]))
        return tf.train.Example(features=tf.train.Features(feature=features))

    writer_option = tf.io.TFRecordOptions(compression_type='GZIP')
    writer = None
    if not tfrecord_size:
        tfrecord_size = df.shape[0]
    for idx, line in tqdm(df.iterrows(), total=df.shape[0]):
        ex = _make_example(line, feature_cols, label_col)
        if idx % tfrecord_size == 0:
            if writer:
                writer.close()
            writer = tf.io.TFRecordWriter(f'{filename_pre}.{idx // tfrecord_size + 1:02d}', options=writer_option)
        writer.write(ex.SerializeToString())


def read_tfrecord(filenames, feature_cols, label_col, batch_size=256, num_epochs=1,
                  num_parallel_calls=8, shuffle_factor=10, prefetch_factor=1):
    """
    :param filenames:
    :param feature_cols:
    :param label_col:
    :param num_epochs: default 1
    :param num_parallel_calls:
    :param shuffle_factor:
    :param prefetch_factor:
    :return:
    """
    feature_description = {feat.name: tf.io.FixedLenFeature([1], dtype=tf.int64) for feat in feature_cols[0]}
    feature_description.update(
        {feat.name: tf.io.FixedLenFeature([1], dtype=tf.float32) for feat in feature_cols[1]})
    feature_description.update(
        {feat.name: tf.io.FixedLenFeature([feat.max_len], dtype=tf.int64) for feat in feature_cols[2]})
    feature_description[label_col['name']] = tf.io.FixedLenFeature([1], dtype=tf.float32)

    def _parse_examples(serial_exmp):
        try:
            features = tf.io.parse_single_example(
                serial_exmp, features=feature_description)
        except AttributeError:
            features = tf.io.parse_single_example(
                serial_exmp, features=feature_description)
        if label_col is not None:
            labels = features.pop(label_col['name'])
            return features, labels
        return features

    def _gen_dataset_iterator():
        filename_list = glob.glob(filenames)
        dataset = tf.data.TFRecordDataset(filename_list, compression_type='GZIP')
        dataset = dataset.map(
            _parse_examples,
            num_parallel_calls=num_parallel_calls)
        if shuffle_factor > 0:
            dataset = dataset.shuffle(buffer_size=batch_size * shuffle_factor)
        dataset = dataset.repeat(num_epochs).batch(batch_size)
        if prefetch_factor > 0:
            dataset = dataset.prefetch(
                buffer_size=batch_size * prefetch_factor)
        try:
            iterator = dataset.make_one_shot_iterator()
        except AttributeError:
            iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        return iterator.get_next()

    return _gen_dataset_iterator()
