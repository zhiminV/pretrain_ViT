import torch
from tqdm import tqdm
from TFtoTorch import test_loader, model

def evaluate_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels.view(-1))
            total_loss += loss.item()
    print(f"Test Loss: {total_loss/len(test_loader):.4f}")

evaluate_model(model, test_loader)
