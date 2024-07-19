import re
from typing import Dict, List, Text, Tuple
import tensorflow as tf
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ViTFeatureExtractor

# 定义输入和输出特征
INPUT_FEATURES = ['elevation', 'th', 'vs', 'tmmn', 'tmmx', 'sph', 'pr', 'pdsi', 'NDVI', 'population', 'erc', 'PrevFireMask']
OUTPUT_FEATURES = ['FireMask', ]

# 数据统计信息
DATA_STATS = {
    'elevation': (0.0, 3141.0, 657.3003, 649.0147),
    'pdsi': (-6.12974870967865, 7.876040384292651, -0.0052714925, 2.6823447),
    'NDVI': (-9821.0, 9996.0, 5157.625, 2466.6677),
    'pr': (0.0, 44.53038024902344, 1.7398051, 4.482833),
    'sph': (0.0, 1.0, 0.0071658953, 0.0042835088),
    'th': (0.0, 360.0, 190.32976, 72.59854),
    'tmmn': (253.15, 298.94891357421875, 281.08768, 8.982386),
    'tmmx': (253.15, 315.09228515625, 295.17383, 9.815496),
    'vs': (0.0, 10.024310074806237, 3.8500874, 1.4109988),
    'erc': (0.0, 106.24891662597656, 37.326267, 20.846027),
    'population': (0.0, 2534.06298828125, 25.531384, 154.72331),
    'PrevFireMask': (-1.0, 1.0, 0.0, 1.0),
    'FireMask': (-1.0, 1.0, 0.0, 1.0),
}

# 数据裁剪和标准化函数
def _clip_and_rescale(inputs: tf.Tensor, key: Text) -> tf.Tensor:
    base_key = re.match(r'([a-zA-Z]+)', key).group(1)
    min_val, max_val, _, _ = DATA_STATS[base_key]
    inputs = tf.clip_by_value(inputs, min_val, max_val)
    return tf.math.divide_no_nan((inputs - min_val), (max_val - min_val))

def _clip_and_normalize(inputs: tf.Tensor, key: Text) -> tf.Tensor:
    base_key = re.match(r'([a-zA-Z]+)', key).group(1)
    min_val, max_val, mean, std = DATA_STATS[base_key]
    inputs = tf.clip_by_value(inputs, min_val, max_val)
    inputs = inputs - mean
    return tf.math.divide_no_nan(inputs, std)

# 数据解析函数
def _parse_fn(example_proto, data_size, sample_size, num_in_channels, clip_and_normalize, clip_and_rescale, random_crop, center_crop):
    input_features, output_features = INPUT_FEATURES, OUTPUT_FEATURES
    feature_names = input_features + output_features
    features_dict = {key: tf.io.FixedLenFeature(shape=[data_size, data_size], dtype=tf.float32) for key in feature_names}
    features = tf.io.parse_single_example(example_proto, features_dict)
    
    inputs_list = [_clip_and_normalize(features[key], key) for key in input_features] if clip_and_normalize else [_clip_and_rescale(features[key], key) for key in input_features]
    inputs_stacked = tf.stack(inputs_list, axis=0)
    input_img = tf.transpose(inputs_stacked, [1, 2, 0])
    
    outputs_list = [features[key] for key in output_features]
    outputs_stacked = tf.stack(outputs_list, axis=0)
    output_img = tf.transpose(outputs_stacked, [1, 2, 0])

    if random_crop:
        input_img, output_img = random_crop_input_and_output_images(input_img, output_img, sample_size, num_in_channels, 1)
    if center_crop:
        input_img, output_img = center_crop_input_and_output_images(input_img, output_img, sample_size)

    return input_img, output_img

# 获取数据集函数
def get_dataset(file_pattern, data_size, sample_size, batch_size, num_in_channels, compression_type, clip_and_normalize, clip_and_rescale, random_crop, center_crop):
    dataset = tf.data.Dataset.list_files(file_pattern)
    dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type=compression_type), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x: _parse_fn(x, data_size, sample_size, num_in_channels, clip_and_normalize, clip_and_rescale, random_crop, center_crop), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

