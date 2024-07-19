from torch.utils.data import Dataset, DataLoader
from transformers import ViTFeatureExtractor
import tensorflow as tf
import torch
from preprocessData import train_dataset,val_dataset,test_dataset

class TFRecordDataset(Dataset):
    def __init__(self, tf_dataset):
        self.data = [x for x in tf_dataset]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        inputs, labels = self.data[idx]
        inputs = torch.from_numpy(inputs.numpy()).permute(2, 0, 1).float()  # 将 (H, W, C) 转换为 (C, H, W)
        labels = torch.from_numpy(labels.numpy()).float()
        return inputs, labels

# 创建 PyTorch 数据集
train_torch_dataset = TFRecordDataset(train_dataset)
val_torch_dataset = TFRecordDataset(val_dataset)
test_torch_dataset = TFRecordDataset(test_dataset)

# 创建 DataLoader
train_loader = DataLoader(train_torch_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_torch_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_torch_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)