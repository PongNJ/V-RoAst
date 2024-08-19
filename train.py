import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import os

from dataloader import MultiTaskDataset
from resnet import MultiTaskResNet
from vgg import MultiTaskVGG

import logging
import os
import argparse



parser = argparse.ArgumentParser(description='train models')
parser.add_argument('model_name', type=str, help='resnet or vgg')

args = parser.parse_args()
model_name = args.model_name

num_classes_per_task = np.load('num_categories.npy')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if model_name == 'resnet':
    log_dir = './resnet_model_results'
    save_folder = "./resnet_model_weight/"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    model = MultiTaskResNet(num_classes_per_task).to(device)

elif model_name == 'vgg':
    log_dir = './VGG_model_results'
    save_folder = "./VGG_model_weight/"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    model = MultiTaskVGG(num_classes_per_task).to(device)

# Set up logging
log_file = os.path.join(log_dir, "training.log")
logging.basicConfig(
    filename=log_file,
    filemode='a',  # 'a' means append, 'w' means overwrite
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Log to both console and file
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Define your image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize the datasets
train_csv_file = './filtered_train_data.csv'
test_csv_file = './filtered_test_data.csv'
val_csv_file = './filtered_val_data.csv'

img_dir = '../traffic_data/ThaiRAP/'

train_dataset = MultiTaskDataset(csv_file=train_csv_file, img_dir=img_dir, transform=transform)
test_dataset = MultiTaskDataset(csv_file=test_csv_file, img_dir=img_dir, transform=transform)
val_dataset = MultiTaskDataset(csv_file=val_csv_file, img_dir=img_dir, transform=transform)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)



# Define loss function for each task
criterion = [nn.CrossEntropyLoss() for _ in range(len(num_classes_per_task))]

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
min_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        # Forward pass
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        # Compute loss for each task and sum them up
        losses = [criterion[i](outputs[i], labels[:, i]) for i in range(len(num_classes_per_task))]
        total_loss = sum(losses)
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}')
    logging.info(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}')
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    correct_preds = [0] * len(num_classes_per_task)
    total_samples = [0] * len(num_classes_per_task)
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            # Compute loss for each task and sum them up
            losses = [criterion[i](outputs[i], labels[:, i]) for i in range(len(num_classes_per_task))]
            total_val_loss = sum(losses)
            
            val_loss += total_val_loss.item()
            
            for i in range(len(num_classes_per_task)):
                _, predicted = torch.max(outputs[i], 1)
                correct_preds[i] += (predicted == labels[:, i]).sum().item()
                total_samples[i] += labels.size(0)
    
    avg_val_loss = val_loss / len(val_loader)


    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')
    logging.info(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')
    
    for i in range(len(num_classes_per_task)):
        accuracy = 100.0 * correct_preds[i] / total_samples[i]
        print(f'Task {i+1} Validation Accuracy: {accuracy:.2f}%')
        logging.info(f'Task {i+1} Validation Accuracy: {accuracy:.2f}%')
    
    if min_val_loss > avg_val_loss:
        model_save_path = os.path.join(save_folder, f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), model_save_path)
        logging.info(f'Model saved at epoch {epoch+1} in {model_save_path}')
        min_val_loss = avg_val_loss
        
logging.info("Training complete!")
print("Training complete!")