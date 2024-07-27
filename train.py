# coding=utf-8
from __future__ import absolute_import, division, print_function
import logging
import torch
import torch.nn as nn
from tqdm import tqdm
from modeling import VisionTransformer, CONFIGS
from data_utils import get_loader
import argparse

logger = logging.getLogger(__name__)


def train_model(model, train_loader, val_loader, epochs=15, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
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

        evaluate_model(model, val_loader)

def evaluate_model(model, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels.view(-1))
            total_loss += loss.item()
    print(f"Validation Loss: {total_loss/len(val_loader):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_loader(args)
    config = CONFIGS["ViT-B_16"]
    model = VisionTransformer(config, img_size=224, num_classes=1)
    train_model(model, train_loader, val_loader)
