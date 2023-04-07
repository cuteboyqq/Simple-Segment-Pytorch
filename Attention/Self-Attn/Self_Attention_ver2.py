# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 13:52:25 2023

@author: User
"""

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialSelfAttention, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, padding=0)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Calculate mean and standard deviation of each channel
        f = self.conv1(x)
        g = self.conv2(x)
        f = f.view(batch_size, channels // 8, -1)
        g = g.view(batch_size, channels // 8, -1)
        
        # Calculate attention map
        attention = torch.bmm(f.permute(0, 2, 1), g)
        attention = torch.softmax(attention, dim=-1)
        
        # Apply attention map to input
        h = x.view(batch_size, channels, -1)
        out = torch.bmm(h, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        
        return out

'''
In this implementation, in_channels is the number of channels in the input feature map, 
and reduction_ratio is the reduction ratio used in the fully connected layers that calculate the attention weights.

The forward method of the ChannelSelfAttention class first applies adaptive average pooling and adaptive max pooling to the input feature map,
 followed by two convolutional layers with ReLU activations to calculate the attention weights.
 It then applies the sigmoid activation function to the sum of the average and max pooling outputs,
 and multiplies this output element-wise with the input feature map to obtain the final output.

You can use this ChannelSelfAttention class in a larger neural network architecture as desired.
'''
class ChannelSelfAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelSelfAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        avg_out = self.fc2(self.relu(self.fc1(avg_out)))
        max_out = self.fc2(self.relu(self.fc1(max_out)))
        out = avg_out + max_out
        out = self.sigmoid(out)
        out = out * x
        
        return out


