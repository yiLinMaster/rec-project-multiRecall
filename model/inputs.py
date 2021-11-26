# -*- coding:utf-8 -*-

import tensorflow as tf
import os
tf.compat.v1.disable_eager_execution()

class InputFn:

    def __init__(self, local_ps, feature_len, batch_size):
        self.feature_len = feature_len
        self.label_len = 1
        self.n_parse_threads = 4
        self.shuffle_buffer_size = 1024
        self.prefetch_buffer_size = 1
        self.batch = batch_size
        self.local_ps = local_ps

    # 读取、解析tfrecords文件；dataset构建、分批+拉取ps中隐向量；返回迭代器
    def input_fn(self, data_dir, is_test = False):
        # 解析单条tfrecords数据
        def _parse_example(example):
            features = {
                "feature": tf.io.FixedLenFeature(self.feature_len, tf.int64),
                "label": tf.io.FixedLenFeature(self.label_len, tf.float32),
            }
            return tf.io.parse_single_example(example, features)

        # 改动2:这里请求参数服务器的单纯隐向量17维(vi,wi)
        # _get_weight代替_get_embedding
        
        def _get_embedding(parsed):
            keys = parsed["feature"]
            keys_array = tf.py_func(
                self.local_ps.pull,[keys],tf.float32
            )
            result = {
                'feature': parsed['feature'],
                'label': parsed['label'],
                'feature_embedding': keys_array
            }
            return result
        
        file_list = os.listdir(data_dir)
        files = []
        for i in range(len(file_list)):
            files.append(os.path.join(data_dir, file_list[i]))

        dataset = tf.data.Dataset.list_files(files)

        if is_test:
            dataset = dataset.repeat(1)
        else:
            dataset = dataset.repeat()

        dataset = dataset.interleave(
            lambda _: tf.data.TFRecordDataset(_),
            cycle_length = 1
        )

        dataset = dataset.map(
            _parse_example,
            num_parallel_calls = self.n_parse_threads
        )

        # 分批
        dataset = dataset.batch(self.batch, drop_remainder = True)

        # 在分batch之后对于每个batch请求参数服务器,因为参数在不断改变
        dataset = dataset.map(_get_embedding, num_parallel_calls = self.n_parse_threads)

        if not is_test:
            dataset.shuffle(self.shuffle_buffer_size)

        dataset = dataset.prefetch(buffer_size = self.prefetch_buffer_size)

        # 返回迭代器
        iterator = tf.data.make_initializable_iterator(dataset)
        return iterator, iterator.get_next()
