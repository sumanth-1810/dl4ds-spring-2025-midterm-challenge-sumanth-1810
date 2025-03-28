#!/usr/bin/env python
# coding: utf-8

# In[1]:


Part 3

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import numpy as np
from tqdm.auto import tqdm
import wandb
from torch.optim.swa_utils import AveragedModel

################################################################################
# Model Definition - Using ResNet152 for Transfer Learning
################################################################################
class ResNet152FineTune(nn.Module):
    def __init__(self):
        super(ResNet152FineTune, self).__init__()
        self.model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)  # Updated weights initialization
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 100)  # Adjust for 100 classes
    
    def forward(self, x):
        return self.model(x)

################################################################################
# Data Augmentation - Using AutoAugment and Mixup
################################################################################
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y, y[index], lam

################################################################################
# Training Function with Mixup
################################################################################
def train(epoch, model, trainloader, optimizer, criterion, CONFIG, scaler):
    device = CONFIG["device"]
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, labels_a, labels_b, lam = mixup_data(inputs, labels)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():  # Mixed precision training
            outputs = model(inputs)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient Clipping
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (lam * predicted.eq(labels_a).sum().item() + (1 - lam) * predicted.eq(labels_b).sum().item())
        
        progress_bar.set_postfix({"loss": running_loss / (total / CONFIG['batch_size']), "acc": 100. * correct / total})
    
    return running_loss / len(trainloader), 100. * correct / total

################################################################################
# Validation Function
################################################################################
def validate(model, valloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(valloader), 100. * correct / total

################################################################################
# Main Function
################################################################################
def main():
    CONFIG = {
        "model": "ResNet152FineTune",
        "batch_size": 128,
        "learning_rate": 7e-4,
        "epochs": 50,
        "num_workers": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge",
        "seed": 42,
    }
    
    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
    model = ResNet152FineTune().to(CONFIG["device"])
    wandb.watch(model)
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    trainset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=True, download=True, transform=transform_train)
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    valloader = torch.utils.data.DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    
    testset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_acc = 0.0
    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG, scaler)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler.step()
        
        wandb.log({"epoch": epoch+1, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")
    
    import eval_cifar100
    import eval_ood
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood3.csv", index=False)
    print("submission_ood3.csv created successfully.")
    
if __name__ == '__main__':
    main()



# In[2]:


pip install -r requirements.txt


# In[ ]:




