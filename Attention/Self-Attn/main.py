# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 01:05:58 2023

@author: User
"""
from Self_Attention import SpatialSelfAttention,ChannelSelfAttention
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.spatial_attention = SpatialSelfAttention(64)
        self.channel_attention = ChannelSelfAttention(64)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.spatial_attention(x)
        x = self.channel_attention(x)
        x = self.conv2(x)
        return x

# Create an instance of the model
model = MyModel()

# Use the model to make a prediction
x = torch.randn(1, 3, 32, 32)
y = model(x)