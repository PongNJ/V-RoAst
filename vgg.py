import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np


class MultiTaskVGG(nn.Module):
    def __init__(self, num_classes_per_task):
        super(MultiTaskVGG, self).__init__()
        
        # Load a pretrained VGG model
        self.vgg = models.vgg16(pretrained=True)  # You can change this to vgg11, vgg19, etc.
        
        # Get the number of input features for the original classifier
        in_features = self.vgg.classifier[-1].in_features
        
        # Replace the classifier's final layer with an identity function
        self.vgg.classifier[-1] = nn.Identity()
        
        # Create a task-specific head for each task
        self.task_heads = nn.ModuleList([nn.Linear(in_features, num_classes) 
                                         for num_classes in num_classes_per_task])

    def forward(self, x):
        features = self.vgg(x)
        
        # Compute output for each task
        outputs = [task_head(features) for task_head in self.task_heads]
        
        return outputs

