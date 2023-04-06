# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 01:36:34 2023

@author: User
"""

import torch
import torch.nn as nn
from fchardnet import FCHarDNet


'''
To use FCHarDNet for semantic segmentation, you can follow these general steps:

1. Define the model architecture by instantiating the FCHarDNet class, specifying the number of input channels and number of classes as arguments.
2. Define the loss function, such as nn.CrossEntropyLoss().
3. Define the optimizer, such as torch.optim.Adam().
4. Train the model by iterating through the training data using a loop, 
   and performing forward and backward passes through the network to optimize the parameters.
5. Evaluate the model on the validation set to monitor the training progress and determine when to stop training.
Here's an example of how you could use FCHarDNet in PyTorch:
'''

'''
Note that in this example, 
train_loader and val_loader are PyTorch DataLoader objects that provide batches of training and validation data. 
You would need to define these loaders based on your own dataset and preprocessing requirements.
'''
# Define the model architecture
num_classes = 10
model = FCHarDNet(in_channels=3, num_classes=num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate the model
with torch.no_grad():
    total_correct = 0
    total_samples = 0
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
    accuracy = total_correct / total_samples
    print('Validation accuracy: {:.2f}%'.format(accuracy * 100))
