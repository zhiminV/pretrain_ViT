import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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

from preprocessData import train_dataset, validation_dataset, test_dataset

train_dataset = TFRecordTorchDataset(train_dataset)
val_dataset = TFRecordTorchDataset(validation_dataset)
test_dataset = TFRecordTorchDataset(test_dataset)

BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
