import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTFeatureExtractor, ViTModel
import torch.nn as nn
from torch.optim import AdamW
from data_utils import train_loader, val_loader
from pretrainViT import model

criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, epochs=15):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

train_model(model, train_loader, val_loader)
