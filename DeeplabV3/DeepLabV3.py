# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 02:26:08 2023

@author: User
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import *
'''
This implementation uses a ResNet-50 backbone and an Atrous Spatial Pyramid Pooling (ASPP) module to extract multi-scale features.
 The ASPP module applies dilated convolutions at different rates to capture context at different scales,
 and then combines the outputs of these convolutions to obtain a final feature representation.
 The final output is obtained by applying a 1x1 convolution to the output of the ASPP module and
 then upsampling the result to the original image size using bilinear interpolation.
'''
'''
Here are some steps you might take to train a semantic segmentation model using this code:

1. Modify the number of classes (n_classes) to match the number of classes in your dataset.

2. Define your loss function, such as cross-entropy loss, which is commonly used for semantic segmentation tasks.

3. Define your optimizer, such as stochastic gradient descent (SGD) or Adam, and specify the learning rate and other hyperparameters.

4. Write a data loader that loads your images and segmentation masks in batches, and preprocesses them as needed (e.g., normalization, resizing, data augmentation).

5. Write a training loop that iterates over the batches of data, computes the loss and gradients, and updates the model parameters using the optimizer.

6. Optionally, evaluate the model on a validation set after each epoch to monitor its performance and prevent overfitting.

Keep in mind that training a semantic segmentation model can be computationally intensive and may require a significant amount of time and resources,
 especially for large datasets.
 It's also important to properly validate your model and tune the hyperparameters to achieve good performance on your specific task.
'''
class DeepLabv3(nn.Module):
    def __init__(self, n_classes=21):
        super(DeepLabv3, self).__init__()

        self.resnet = torchvision.models.resnet50(pretrained=True)

        self.aspp = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=8, dilation=8),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=16, dilation=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, kernel_size=1),
        )

    def forward(self, x):
        size = x.size()[2:]
        features = self.resnet.features(x)
        x = self.aspp(features)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        logits = x
        probabilities = F.softmax(x, dim=1)
        return logits, probabilities
