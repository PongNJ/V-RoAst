import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np


class MultiTaskResNet(nn.Module):
    def __init__(self, num_classes_per_task):
        super(MultiTaskResNet, self).__init__()
        
        # Load a pretrained ResNet model
        self.resnet = models.resnet18(pretrained=False)
        
        # Get the number of input features for the original fully connected layer
        in_features = self.resnet.fc.in_features
        
        # Replace the original fully connected layer with an identity function
        self.resnet.fc = nn.Identity()
        
        # Create a task-specific head for each task
        self.task_heads = nn.ModuleList([nn.Linear(in_features, num_classes) 
                                         for num_classes in num_classes_per_task])

    def forward(self, x):
        features = self.resnet(x)
        
        # Compute output for each task
        outputs = [task_head(features) for task_head in self.task_heads]
        
        return outputs
    
