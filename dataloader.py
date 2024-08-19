import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

# class MultiTaskDataset(Dataset):
#     def __init__(self, csv_file, img_dir, transform=None):
#         self.annotations = pd.read_csv(csv_file)
#         self.img_dir = img_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, idx):
#         # Get the image ID and construct the image file name
#         img_id = str(self.annotations.iloc[idx, 0])
#         img_name = img_id + '.jpg'
#         img_path = os.path.join(self.img_dir, img_name)
        
#         # Open the image
#         image = Image.open(img_path).convert("RGB")
        
#         # Get all labels for the tasks
#         labels = self.annotations.iloc[idx, 1:].values.astype(int)
#         labels = torch.tensor(labels, dtype=torch.long)

#         # Apply transformations if specified
#         if self.transform:
#             image = self.transform(image)

#         return image, labels

class MultiTaskDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get the image ID and construct the image file name
        img_id = str(self.annotations.iloc[idx, 0])
        img_name = img_id + '.jpg'
        img_path = os.path.join(self.img_dir, img_name)
        
        # Open the image
        image = Image.open(img_path).convert("RGB")
        
        # Get all labels for the tasks and adjust them
        labels = self.annotations.iloc[idx, 1:].values.astype(int)
        labels = torch.tensor(labels - 1, dtype=torch.long)  # Adjust labels from 1,2 to 0,1

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        return image, labels, img_id


# if __name__== '__main__':
#     from torchvision import transforms

#     # Define your image transformations
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])

#     # Paths to your data
#     csv_file = '../traffic_data/final_train_data.csv'
#     img_dir = '../traffic_data/ThaiRAP/'

#     # Create the dataset
#     dataset = MultiTaskDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)

#     # Create the DataLoader
#     train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)