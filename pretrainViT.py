from transformers import ViTForImageClassification, AdamW, get_scheduler
from TFtoTorch import train_loader,val_loader,test_loader
import tensorflow as tf
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=1)
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = 20 * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

criterion = torch.nn.BCEWithLogitsLoss()

# 训练循环
model.train()
for epoch in range(20):
    for batch in train_loader:
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs.logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # 验证
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            val_loss += criterion(outputs.logits, labels).item()
            pred = outputs.logits > 0.5
            correct += pred.eq(labels.view_as(pred)).sum().item()
    
    val_loss /= len(val_loader.dataset)
    val_accuracy = correct / len(val_loader.dataset)
    
    print(f"Validation loss: {val_loss}, Validation accuracy: {val_accuracy}")
    model.train()

# 测试模型
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        test_loss += criterion(outputs.logits, labels).item()
        pred = outputs.logits > 0.5
        correct += pred.eq(labels.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
test_accuracy = correct / len(test_loader.dataset)

print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")