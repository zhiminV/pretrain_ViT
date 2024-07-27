import logging
import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torchvision import transforms
import tensorflow as tf
from typing import Dict, List, Text, Tuple
from preprocessData import get_dataset

logger = logging.getLogger(__name__)

class NextDayFireDataset(torch.utils.data.Dataset):
    def __init__(self, tf_dataset, transform=None):
        self.tf_dataset = list(tf_dataset)
        self.transform = transform

    def __len__(self):
        return len(self.tf_dataset)

    def __getitem__(self, idx):
        image, label = self.tf_dataset[idx]
        if self.transform:
            image = self.transform(image.numpy())
        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1), torch.tensor(label.numpy(), dtype=torch.float32)

def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

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

    train_dataset_tf = get_dataset('/Users/lzm/Desktop/7980 Capstone/rayan 项目/northamerica_2012-2023/train/*_ongoing_*.tfrecord', data_size=64, sample_size=224, batch_size=args.train_batch_size, num_in_channels=12, compression_type=None, clip_and_normalize=True, clip_and_rescale=False, random_crop=True, center_crop=False)
    val_dataset_tf = get_dataset('/Users/lzm/Desktop/7980 Capstone/rayan 项目/northamerica_2012-2023/val/*_ongoing_*.tfrecord', data_size=64, sample_size=224, batch_size=args.eval_batch_size, num_in_channels=12, compression_type=None, clip_and_normalize=True, clip_and_rescale=False, random_crop=True, center_crop=False)
    test_dataset_tf = get_dataset('/Users/lzm/Desktop/7980 Capstone/rayan 项目/northamerica_2012-2023/test/*_ongoing_*.tfrecord', data_size=64, sample_size=224, batch_size=args.eval_batch_size, num_in_channels=12, compression_type=None, clip_and_normalize=True, clip_and_rescale=False, random_crop=True, center_crop=False)

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
