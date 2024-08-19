import pandas as pd
import re
import argparse
import os
import numpy as np
# Task names
tasks = ['Carriageway', 'Upgrade cost', 'Motorcycle observed flow',
       'Pedestrian observed flow along the road driver-side',
       'Pedestrian observed flow along the road passenger-side',
       'Land use - driver-side', 'Land use - passenger-side', 'Area type',
       'Speed limit', 'Motorcycle speed limit', 'Truck speed limit',
       'Differential speed limits', 'Median type',
       'Roadside severity - driver-side distance',
       'Roadside severity - driver-side object',
       'Roadside severity - passenger-side distance',
       'Roadside severity - passenger-side object', 'Shoulder rumble strips',
       'Paved shoulder - driver-side', 'Paved shoulder - passenger-side',
       'Intersection type', 'Intersection channelisation',
       'Intersecting road volume', 'Intersection quality',
       'Property access points', 'Number of lanes', 'Curvature',
       'Quality of curve', 'Skid resistance / grip', 'Delineation',
       'Street lighting', 'Pedestrian crossing facilities - inspected road',
       'Pedestrian crossing quality',
       'Pedestrian crossing facilities - intersecting road',
       'Speed management / traffic calming', 'Pedestrian fencing',
       'Vehicle parking', 'Sidewalk - passenger-side', 'Service road',
       'School zone warning', 'School zone crossing supervisor']


parser = argparse.ArgumentParser(description='evaluate models')
parser.add_argument('model_name', type=str, help='resnet or vgg')
parser.add_argument('dataset', type=str, help='test or unseen')

args = parser.parse_args()
model_name = args.model_name
dataset_type = args.dataset

if model_name == 'resnet':
    log_dir = './resnet_model_results'
    if dataset_type == 'test':
        log_file_path = os.path.join(log_dir, "test_evaluation.log")
    elif dataset_type == 'unseen':
        log_file_path = os.path.join(log_dir, "unseen_evaluation.log")

if model_name == 'vgg':
    log_dir = './VGG_model_results'
    if dataset_type == 'test':
        log_file_path = os.path.join(log_dir, "test_evaluation.log")
    elif dataset_type == 'unseen':
        log_file_path = os.path.join(log_dir, "unseen_evaluation.log")



# Load log file content from disk
with open(log_file_path, 'r') as file:
    log_file = file.read()

num_classes_per_task = np.load('num_categories.npy')

# Extracting the data
data = {}
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

for task_num in range(1, len(num_classes_per_task)+1):
    task_data = {}
    for metric in metrics:
        pattern = fr"Task {task_num} {metric}: ([\d.]+)%?"
        match = re.search(pattern, log_file)
        if match:
            task_data[metric] = float(match.group(1))
    data[tasks[task_num - 1]] = task_data

# Creating DataFrame
df = pd.DataFrame(data).T

if model_name == 'resnet':
    if dataset_type == 'test':
        output_file_path = "./resnet_model_results/test_output.csv"
        df.to_csv(output_file_path, index=True)
    elif dataset_type == 'unseen':
        output_file_path = "./resnet_model_results/unseen_output.csv"
        df.to_csv(output_file_path, index=True)
if model_name == 'vgg':
    if dataset_type == 'test':
        output_file_path = "./VGG_model_results/test_output.csv"
        df.to_csv(output_file_path, index=True)
    elif dataset_type == 'unseen':
        output_file_path = "./VGG_model_results/unseen_output.csv"
        df.to_csv(output_file_path, index=True)