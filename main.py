# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 01:20:06 2023

@author: User
"""

'''
In this example,
 we first create an instance of the UNet model with n_channels=3 and n_classes=2. 
 We then load an input image with shape (1, 3, 256, 256), 
 representing one image with three color channels and a spatial resolution of 256x256 pixels.
 We pass the image through the UNet model to get the predicted class probabilities with shape (1, 2, 256, 256),
 representing one image with two classes (background and foreground) and a spatial resolution of 256x256 pixels.
 Finally, we convert the output to a segmentation mask by taking the argmax along the channel dimension to get a tensor with shape (1, 256, 256),
 representing the predicted class label for each pixel in the input image.
'''


import torch
from unet import UNet

# Create an instance of the UNet model
model = UNet(n_channels=3, n_classes=2)

# Load an input image
image = torch.randn((1, 3, 256, 256))

# Pass the image through the model to get the predicted class probabilities
output = model(image)

# Convert the output to a segmentation mask by taking the argmax along the channel dimension
segmentation_mask = torch.argmax(output, dim=1)

