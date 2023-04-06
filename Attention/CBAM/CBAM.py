# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
'''
Here's an implementation of the Convolutional Block Attention Module (CBAM) in PyTorch:
'''

'''
This implementation defines two sub-modules: 
    a channel attention module (ChannelAttention) and a spatial attention module (SpatialAttention).
    The CBAM module combines these two sub-modules to perform both channel-wise and spatial attention on the input feature maps.

The ChannelAttention module performs global average pooling and global max pooling on the input feature maps,
 followed by two fully connected layers with ReLU activation and a sigmoid activation.
 The outputs of the fully connected layers are added element-wise and then passed through the sigmoid function to generate a channel-wise attention map.

The SpatialAttention module performs channel-wise average pooling and channel-wise max pooling on the input feature maps,
 concatenates the outputs along the channel dimension,
 and passes the concatenated feature maps through a convolutional layer followed by a sigmoid activation to generate a spatial attention map.

The CBAM module applies both the channel-wise and spatial attention maps to the input feature maps and returns the result.
'''
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_gate = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_gate = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_gate(x)
        x = x * self.spatial_gate(x)
        return x


