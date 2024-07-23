import re
from typing import Dict, List, Text, Tuple
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

INPUT_FEATURES = ['elevation', 'th', 'vs', 'tmmn', 'tmmx', 'sph', 'pr', 'pdsi', 'NDVI', 'population', 'erc', 'PrevFireMask']
OUTPUT_FEATURES = ['FireMask', ]

DATA_STATS = {
    'elevation': (0.0, 3141.0, 657.3003, 649.0147),
    'pdsi': (-6.12974870967865, 7.876040384292651, -0.0052714925, 2.6823447),
    'NDVI': (-9821.0, 9996.0, 5157.625, 2466.6677),
    'pr': (0.0, 44.53038024902344, 1.7398051, 4.482833),
    'sph': (0., 1., 0.0071658953, 0.0042835088),
    'th': (0., 360.0, 190.32976, 72.59854),
    'tmmn': (253.15, 298.94891357421875, 281.08768, 8.982386),
    'tmmx': (253.15, 315.09228515625, 295.17383, 9.815496),
    'vs': (0.0, 10.024310074806237, 3.8500874, 1.4109988),
    'erc': (0.0, 106.24891662597656, 37.326267, 20.846027),
    'population': (0., 2534.06298828125, 25.531384, 154.72331),
    'PrevFireMask': (-1., 1., 0., 1.),
    'FireMask': (-1., 1., 0., 1.)
}

def random_crop_input_and_output_images(input_img: tf.Tensor, output_img: tf.Tensor, sample_size: int, num_in_channels: int, num_out_channels: int) -> Tuple[tf.Tensor, tf.Tensor]:
    combined = tf.concat([input_img, output_img], axis=2)
    combined = tf.image.random_crop(combined, [sample_size, sample_size, num_in_channels + num_out_channels])
    input_img = combined[:, :, 0:num_in_channels]
    output_img = combined[:, :, -num_out_channels:]
    return input_img, output_img

def center_crop_input_and_output_images(input_img: tf.Tensor, output_img: tf.Tensor, sample_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
    central_fraction = sample_size / input_img.shape[0]
    input_img = tf.image.central_crop(input_img, central_fraction)
    output_img = tf.image.central_crop(output_img, central_fraction)
    return input_img, output_img

def _get_base_key(key: Text) -> Text:
    match = re.match(r'([a-zA-Z]+)', key)
    if match:
        return match.group(1)
    raise ValueError('The provided key does not match the expected pattern: {}'.format(key))

def _clip_and_rescale(inputs: tf.Tensor, key: Text) -> tf.Tensor:
    base_key = _get_base_key(key)
    if base_key not in DATA_STATS:
        raise ValueError('No data statistics available for the requested key: {}.'.format(key))
    min_val, max_val, _, _ = DATA_STATS[base_key]
    inputs = tf.clip_by_value(inputs, min_val, max_val)
    return tf.math.divide_no_nan((inputs - min_val), (max_val - min_val))

def _clip_and_normalize(inputs: tf.Tensor, key: Text) -> tf.Tensor:
    base_key = _get_base_key(key)
    if base_key not in DATA_STATS:
        raise ValueError('No data statistics available for the requested key: {}.'.format(key))
    min_val, max_val, mean, std = DATA_STATS[base_key]
    inputs = tf.clip_by_value(inputs, min_val, max_val)
    inputs = inputs - mean
    return tf.math.divide_no_nan(inputs, std)

def _get_features_dict(sample_size: int, features: List[Text]) -> Dict[Text, tf.io.FixedLenFeature]:
    sample_shape = [sample_size, sample_size]
    features = set(features)
    columns = [tf.io.FixedLenFeature(shape=sample_shape, dtype=tf.float32) for _ in features]
    return dict(zip(features, columns))

def _parse_fn(example_proto: tf.train.Example, data_size: int, sample_size: int, num_in_channels: int, clip_and_normalize: bool, clip_and_rescale: bool, random_crop: bool, center_crop: bool) -> Tuple[tf.Tensor, tf.Tensor]:
    if (random_crop and center_crop):
        raise ValueError('Cannot have both random_crop and center_crop be True')
    input_features, output_features = INPUT_FEATURES, OUTPUT_FEATURES
    feature_names = input_features + output_features
    features_dict = _get_features_dict(data_size, feature_names)
    features = tf.io.parse_single_example(example_proto, features_dict)

    if clip_and_normalize:
        inputs_list = [_clip_and_normalize(features.get(key), key) for key in input_features]
    elif clip_and_rescale:
        inputs_list = [_clip_and_rescale(features.get(key), key) for key in input_features]
    else:
        inputs_list = [features.get(key) for key in input_features]
  
    inputs_stacked = tf.stack(inputs_list, axis=0)
    input_img = tf.transpose(inputs_stacked, [1, 2, 0])

    outputs_list = [features.get(key) for key in output_features]
    assert outputs_list, 'outputs_list should not be empty'
    outputs_stacked = tf.stack(outputs_list, axis=0)

    outputs_stacked_shape = outputs_stacked.get_shape().as_list()
    assert len(outputs_stacked.shape) == 3, ('outputs_stacked should be rank 3'
                                            'but dimensions of outputs_stacked'
                                            f' are {outputs_stacked_shape}')
    output_img = tf.transpose(outputs_stacked, [1, 2, 0])

    if random_crop:
        input_img, output_img = random_crop_input_and_output_images(input_img, output_img, sample_size, num_in_channels, 1)
    if center_crop:
        input_img, output_img = center_crop_input_and_output_images(input_img, output_img, sample_size)
    return input_img, output_img

def get_dataset(file_pattern: Text, data_size: int, sample_size: int, batch_size: int, num_in_channels: int, compression_type: Text, clip_and_normalize: bool, clip_and_rescale: bool, random_crop: bool, center_crop: bool) -> tf.data.Dataset:
    if (clip_and_normalize and clip_and_rescale):
        raise ValueError('Cannot have both normalize and rescale.')
    dataset = tf.data.Dataset.list_files(file_pattern)
    dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type=compression_type), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x: _parse_fn(x, data_size, sample_size, num_in_channels, clip_and_normalize, clip_and_rescale, random_crop, center_crop), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

BATCH_SIZE = 32

train_dataset = get_dataset('/home/liang.zhimi/ondemand/northamerica_2012-2023/train/*_ongoing_*.tfrecord', data_size=64, sample_size=32, batch_size=BATCH_SIZE, num_in_channels=12, compression_type=None, clip_and_normalize=True, clip_and_rescale=False, random_crop=True, center_crop=False)

validation_dataset = get_dataset('/home/liang.zhimi/ondemand/northamerica_2012-2023/val/*_ongoing_*.tfrecord', data_size=64, sample_size=32, batch_size=BATCH_SIZE, num_in_channels=12, compression_type=None, clip_and_normalize=True, clip_and_rescale=False, random_crop=True, center_crop=False)

test_dataset = get_dataset('/home/liang.zhimi/ondemand/northamerica_2012-2023/test/*_ongoing_*.tfrecord', data_size=64, sample_size=32, batch_size=BATCH_SIZE, num_in_channels=12, compression_type=None, clip_and_normalize=True, clip_and_rescale=False, random_crop=True, center_crop=False)


# def count_elements(dataset):
#     count = 0
#     for _ in dataset:
#         count += 1
#     return count

# print(f"Train dataset size: {count_elements(train_dataset)}")
# print(f"Validation dataset size: {count_elements(validation_dataset)}")
# print(f"Test dataset size: {count_elements(test_dataset)}")

# rain dataset size: 993
# Validation dataset size: 357
# Test dataset size: 257