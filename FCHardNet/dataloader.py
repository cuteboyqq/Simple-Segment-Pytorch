# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 01:43:53 2023

@author: User
"""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
'''
In this example, 
image_paths and label_paths are lists of file paths to the image and label map files,
 respectively. Inside __getitem__,
 we load the image and label map files using OpenCV and apply any necessary preprocessing steps.
 We also convert the label map to a one-hot encoded tensor for use with the cross-entropy loss function.
 Finally, we return the preprocessed image and one-hot encoded label map as a tuple.
'''

# Define your custom dataset class
class MyDataset(Dataset):
    def __init__(self, image_paths, label_paths):
        self.image_paths = image_paths
        self.label_paths = label_paths

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = self.label_paths[index]

        # Load the image and label map
        image = cv2.imread(image_path)
        label_map = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        # Apply any necessary preprocessing to the image and label map
        image = ...
        label_map = ...

        # Convert label map to tensor and apply one-hot encoding
        label_map_tensor = torch.from_numpy(label_map).long()
        label_map_onehot = torch.zeros(num_classes, label_map_tensor.size(0), label_map_tensor.size(1))
        label_map_onehot.scatter_(0, label_map_tensor.unsqueeze(0), 1)

        return image, label_map_onehot

    def __len__(self):
        return len(self.image_paths)

# Define your train and validation datasets
train_data = ...
train_targets = ...
val_data = ...
val_targets = ...

# Define your train and validation data loaders
batch_size = 32
train_dataset = MyDataset(train_data, train_targets)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = MyDataset(val_data, val_targets)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
