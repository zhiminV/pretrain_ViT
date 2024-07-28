import logging
import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torchvision import transforms
import tensorflow as tf
from typing import Dict, List, Text, Tuple, Literal
from preprocessData import get_dataset
import numpy as np
from PIL import Image
from preprocessData import (
    INPUT_FEATURES,
    OUTPUT_FEATURES,
    _get_features_dict,
    _clip_and_normalize,
    calculate_fire_change,
    random_crop,
)
logger = logging.getLogger(__name__)

class NextDayFireDataset(torch.utils.data.Dataset):
    """Next Day Fire dataset."""

    def __init__(
        self,
        tf_dataset: tf.data.Dataset,
        transform=None,
        clip_normalize: bool = True,
        limit_features_list: list = None,
        use_change_mask: bool = False,
        sampling_method: Literal[
            'random_crop', 'center_crop', 'downsample', 'original'
        ] = 'random_crop',
        mode: Literal['train', 'val', 'test'] = 'train',
    ):
        self.tf_dataset = tf_dataset
        self.transform = transform
        self.feature_description = _get_features_dict(
            64, INPUT_FEATURES + OUTPUT_FEATURES
        )
        self.mask_values = [0, 1]  # only 0 and 1 are valid mask values
        self.clip_normalize = clip_normalize
        self.use_change_mask = use_change_mask
        self.sampling_method = sampling_method
        self.mode = mode
        self.limit_features_list = limit_features_list
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return self.tf_dataset.reduce(0, lambda x, _: x + 1).numpy()

    def __getitem__(self, idx):
        item = next(iter(self.tf_dataset.skip(idx).take(1)))
        item = tf.io.parse_single_example(item, self.feature_description)
        target = item.pop('FireMask')
        # Get the change mask
        if self.use_change_mask:
            target = calculate_fire_change(item.get('PrevFireMask'), target)
        if self.limit_features_list:
            item = {key: item[key] for key in self.limit_features_list}
            # Clip and normalize features
            if self.clip_normalize:
                item = [
                    _clip_and_normalize(item.get(key), key)
                    for key in self.limit_features_list
                ]
            else:
                item = [item.get(key) for key in self.limit_features_list]

        if not self.use_change_mask:
            target = tf.cast(target, tf.float16).numpy()
            if self.mode == 'train':
                # convert to binary mask
                target = np.where(target > 0, 1, 0)
        target = np.expand_dims(target, axis=0)
        features = [tf.cast(x, tf.float16).numpy() for x in list(item.values())]
        item = np.stack(features, axis=0)

        if self.sampling_method == 'random_crop':
            item, target = random_crop(
                item,
                target,
            )
        elif self.sampling_method == 'center_crop':
            item = item[:, 16:-16, 16:-16]
            target = target[:, 16:-16, 16:-16]
        elif self.sampling_method == 'downsample':
            item = item[:, ::2, ::2]
            target = target[:, ::2, ::2]
        elif self.sampling_method == 'original':
            pass
        else:
            raise NotImplementedError

        # Transform the features to a PIL Image and apply transforms
        item = np.moveaxis(item, 0, -1)  # Move channels to the last dimension
        item = Image.fromarray((item * 255).astype('uint8'))

        if self.transform:
            item = self.transform(item)

        return item, target


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # Adjust the crop size to match the input image size
    crop_size = 32  # or any size <= 64

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset_tf = get_dataset('/home/liang.zhimi/ondemand/northamerica_2012-2023/train/*_ongoing_*.tfrecord', data_size=64, sample_size=crop_size, batch_size=args.train_batch_size, num_in_channels=12, compression_type=None, clip_and_normalize=True, clip_and_rescale=False, random_crop=True, center_crop=False)
    val_dataset_tf = get_dataset('/home/liang.zhimi/ondemand/northamerica_2012-2023/val/*_ongoing_*.tfrecord', data_size=64, sample_size=crop_size, batch_size=args.eval_batch_size, num_in_channels=12, compression_type=None, clip_and_normalize=True, clip_and_rescale=False, random_crop=True, center_crop=False)
    test_dataset_tf = get_dataset('/home/liang.zhimi/ondemand/northamerica_2012-2023/test/*_ongoing_*.tfrecord', data_size=64, sample_size=crop_size, batch_size=args.eval_batch_size, num_in_channels=12, compression_type=None, clip_and_normalize=True, clip_and_rescale=False, random_crop=True, center_crop=False)

    train_dataset = NextDayFireDataset(train_dataset_tf, transform=transform_train)
    val_dataset = NextDayFireDataset(val_dataset_tf, transform=transform_test)
    test_dataset = NextDayFireDataset(test_dataset_tf, transform=transform_test)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    test_sampler = SequentialSampler(test_dataset)

    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=4, pin_memory=True) if test_dataset is not None else None

    return train_loader, val_loader, test_loader
