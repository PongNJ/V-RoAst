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
from sklearn.metrics import precision_score, recall_score, f1_score
import argparse

parser = argparse.ArgumentParser(description='evaluate models')
parser.add_argument('model_name', type=str, help='resnet or vgg')
parser.add_argument('dataset', type=str, help='test or unseen')
parser.add_argument('task', type=str, help='inference or each')

args = parser.parse_args()
model_name = args.model_name
dataset_type = args.dataset
task_type = args.task


num_classes_per_task = np.load('num_categories.npy')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if model_name == 'resnet':
    log_dir = './resnet_model_results'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if task_type == 'inference':
        batch_size = 32
        if dataset_type == 'test':
            log_file = os.path.join(log_dir, "test_evaluation.log")
        elif dataset_type == 'unseen':
            log_file = os.path.join(log_dir, "unseen_evaluation.log")
    elif task_type == 'each':
        if dataset_type == 'test':
            log_file = os.path.join(log_dir, "each_test_data_evaluation.log")
        elif dataset_type == 'unseen':
            log_file = os.path.join(log_dir, "each_unseen_data_evaluation.log")
        batch_size = 1
    model = MultiTaskResNet(num_classes_per_task)

elif model_name == 'vgg':
    log_dir = './VGG_model_results'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if task_type == 'inference':
        batch_size = 32
        if dataset_type == 'test':
            log_file = os.path.join(log_dir, "test_evaluation.log")
        elif dataset_type == 'unseen':
            log_file = os.path.join(log_dir, "unseen_evaluation.log")
    elif task_type == 'each':
        if dataset_type == 'test':
            log_file = os.path.join(log_dir, "each_test_data_evaluation.log")
        elif dataset_type == 'unseen':
            log_file = os.path.join(log_dir, "each_unseen_data_evaluation.log")
        batch_size = 1
    model = MultiTaskVGG(num_classes_per_task)

# Set up logging
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
test_csv_file = './final_dataset/filtered_test_data.csv'
unseen_csv_file = './final_dataset/unseen_data.csv'
img_dir = '../traffic_data/ThaiRAP/'

if dataset_type == 'test':
    test_dataset = MultiTaskDataset(csv_file=test_csv_file, img_dir=img_dir, transform=transform)
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

elif dataset_type == 'unseen':
    unseen_dataset = MultiTaskDataset(csv_file=unseen_csv_file, img_dir=img_dir, transform=transform)
    dataloader = DataLoader(unseen_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

if model_name == 'resnet':
    model.load_state_dict(torch.load('./resnet_model_weight/model_epoch_25.pth'))
elif model_name == 'vgg':
    model.load_state_dict(torch.load('./VGG_model_weight/model_epoch_17.pth'))
model.to(device)

correct_preds = [0] * len(num_classes_per_task)  # To store correct predictions for each task
total_samples = [0] * len(num_classes_per_task)  # To store total samples for each task

# Lists to store metrics for each task
precisions, recalls, f1_scores = [], [], []
# Set the model to evaluation mode
model.eval()
with torch.no_grad(): 
    for images, labels, id in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Compute accuracy for each task
        if task_type == 'inference':
            for i in range(len(num_classes_per_task)):
                _, predicted = torch.max(outputs[i], 1)
                correct_preds[i] += (predicted == labels[:, i]).sum().item()
                total_samples[i] += labels.size(0)
                precisions.append(precision_score(labels[:, i].cpu(), predicted.cpu(), average='macro'))
                recalls.append(recall_score(labels[:, i].cpu(), predicted.cpu(), average='macro'))
                f1_scores.append(f1_score(labels[:, i].cpu(), predicted.cpu(), average='macro'))
                
        elif task_type == 'each':
            for i in range(len(num_classes_per_task)):
                _, predicted = torch.max(outputs[i], 1)
                logging.info(f'Image ID: {id[0]} | Predicted: {predicted.item()+1} | Actual: {labels[:, i].item()+1}')
                logging.info('-----------------------------------')
                
if task_type == 'inference':
    # Calculate accuracy for each task
    for i in range(len(num_classes_per_task)):
        accuracy = 100.0 * correct_preds[i] / total_samples[i]
        logging.info(f'Task {i+1} Accuracy: {accuracy:.2f}%')
        logging.info(f'Task {i+1} Precision: {precisions[i]:.2f}')
        logging.info(f'Task {i+1} Recall: {recalls[i]:.2f}')
        logging.info(f'Task {i+1} F1-Score: {f1_scores[i]:.2f}')

    logging.info("Evaluation complete!")
    
    