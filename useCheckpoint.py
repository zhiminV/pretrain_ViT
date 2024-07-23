import re
import os
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import ViTFeatureExtractor, ViTModel
from sklearn.metrics import recall_score, f1_score
from typing import Dict, List, Text, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors

# Define input and output features
INPUT_FEATURES = ['elevation', 'th', 'vs', 'tmmn', 'tmmx', 'sph', 'pr', 'pdsi', 'NDVI', 'population', 'erc', 'PrevFireMask']
OUTPUT_FEATURES = ['FireMask', ]

# Define dataset stats
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

# Define model class
class ViTForWildfireSpread(nn.Module):
    def __init__(self, num_labels=1):
        super(ViTForWildfireSpread, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(768, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, num_labels, kernel_size=1)

    def forward(self, x):
        x = self.vit(pixel_values=x).last_hidden_state
        x = x.permute(0, 2, 1).reshape(x.shape[0], 768, 14, 14)
        x = self.upsample(x)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ViTForWildfireSpread().to(device)

# Load the pre-trained weights
weights = np.load('ViT-B_16.npz')
model.load_state_dict(torch.load(weights, map_location=device), strict=False)

# Define dataset classes and helper functions
class TFRecordTorchDataset(Dataset):
    def __init__(self, tf_dataset):
        self.tf_dataset = tf_dataset
        self.tf_iter = iter(tf_dataset)
    
    def __len__(self):
        return sum(1 for _ in self.tf_dataset)
    
    def __getitem__(self, idx):
        try:
            data = next(self.tf_iter)
        except StopIteration:
            self.tf_iter = iter(self.tf_dataset)
            data = next(self.tf_iter)
        image, label = data
        return torch.tensor(image.numpy(), dtype=torch.float32), torch.tensor(label.numpy(), dtype=torch.long)

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

train_dataset = TFRecordTorchDataset(train_dataset)
val_dataset = TFRecordTorchDataset(validation_dataset)
test_dataset = TFRecordTorchDataset(test_dataset)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

# Evaluate the model
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Evaluating"):
        inputs, labels = inputs.to(device), labels.to(device).float()
        outputs = model(inputs)
        preds = torch.round(torch.sigmoid(outputs)).cpu().numpy()
        all_preds.extend(preds.flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
print(f"Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# Visualize the results
def show_inference(n_rows, features, label, prediction_function):
    CMAP = colors.ListedColormap(['black', 'silver', 'orangered'])
    BOUNDS = [-1, -0.1, 0.001, 1]
    NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)
    
    fig = plt.figure(figsize=(15, n_rows * 4))
    
    features = features.permute(0, 2, 3, 1)  # Change feature tensor to (batch, height, width, channels)
    prediction = prediction_function(features)

    for i in range(n_rows):
        plt.subplot(n_rows, 3, i * 3 + 1)
        plt.title("Previous day fire")
        plt.imshow(features[i, :, :, -1].cpu().numpy(), cmap=CMAP, norm=NORM)
        plt.axis('off')

        plt.subplot(n_rows, 3, i * 3 + 2)
        plt.title("True next day fire")
        plt.imshow(label[i, 0, :, :].cpu().numpy(), cmap=CMAP, norm=NORM)
        plt.axis('off')

        plt.subplot(n_rows, 3, i * 3 + 3)
        plt.title("Predicted next day fire")
        plt.imshow(prediction[i, 0, :, :], cmap=CMAP, norm=NORM)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('inference_results.png')
    plt.show()

features, labels = next(iter(test_loader))
features_torch = features.permute(0, 3, 1, 2).float().to(device)
labels_torch = labels.permute(0, 3, 1, 2).float().to(device)
show_inference(5, features_torch, labels_torch, lambda x: torch.sigmoid(model(x.permute(0, 3, 1, 2))).detach().cpu().numpy())
