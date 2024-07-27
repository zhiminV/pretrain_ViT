import torch
import preprocessData
import pretrainViT
import data_utils
import train
import evaluation
import showOutput
import re
from typing import Dict, List, Optional, Text, Tuple
import matplotlib.pyplot as plt
from matplotlib import colors
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils.losses_utils import reduce_weighted_loss
from transformers import ViTFeatureExtractor, ViTModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os


def main():
    # Ensure all necessary imports and settings are in place
    from dataset_wrapper import train_loader, val_loader, test_loader
    from model_definition import model, device

    # Train the model
    print("Starting training...")
    train.train_model(model, train_loader, val_loader, epochs=15)

    # Evaluate the model
    print("Starting evaluation...")
    evaluation.evaluate_model(model, test_loader)

    # Run inference and show results
    print("Running inference and visualizing results...")
    features, labels = next(iter(test_loader))
    features_torch = features.permute(0, 3, 1, 2).float().to(device)
    labels_torch = labels.permute(0, 3, 1, 2).float().to(device)
    inference.show_inference(5, features_torch, labels_torch, lambda x: torch.sigmoid(model(x.permute(0, 3, 1, 2))).detach().cpu().numpy())

if __name__ == "__main__":
    main()
