from __future__ import absolute_import, division, print_function
import logging
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from modeling import VisionTransformer, CONFIGS
from data_utils import get_loader
import argparse
from transformers import AdamW 

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_model(model, train_loader, val_loader, epochs=15, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        print(f"Starting Epoch {epoch+1}")

        # Debug data loading
        for i, (inputs, labels) in enumerate(train_loader):
            print(f"Batch {i+1} loaded.")
            if i == 2:  # Load a few batches and then break
                break
        
        # Training loop
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/15")):
            print(f"Batch {i+1}/{len(train_loader)} loaded successfully.")
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"Batch {i+1}/{len(train_loader)}, Loss: {loss.item()}")
        
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "your_dataset_name"], required=True)
    parser.add_argument("--model_type", choices=list(CONFIGS.keys()), required=True)
    parser.add_argument("--pretrained_dir", type=str, required=True)
    parser.add_argument("--train_batch_size", default=16, type=int)  # Reduced batch size
    parser.add_argument("--eval_batch_size", default=16, type=int)    # Reduced batch size
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fp16_opt_level", type=str, default="O1")
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_loader(args)
    config = CONFIGS[args.model_type]
    model = VisionTransformer(config, num_classes=1, zero_head=True)

    model.load_from(np.load(args.pretrained_dir))
    train_model(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
