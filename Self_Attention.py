# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 01:03:15 2023

@author: User
"""

import torch
import torch.nn as nn

class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # Input x has shape (batch_size, channels, height, width)
        
        # Compute queries, keys, and values
        queries = self.conv(x)
        keys = self.conv(x)
        values = x
        
        # Reshape queries, keys, and values
        batch_size, channels, height, width = x.size()
        queries = queries.view(batch_size, channels, height*width)
        keys = keys.view(batch_size, channels, height*width)
        values = values.view(batch_size, channels, height*width)
        
        # Compute dot product attention scores
        scores = torch.bmm(queries.transpose(1,2), keys)
        scores = scores / (height*width)
        scores = self.softmax(scores)
        
        # Apply attention to values
        attn_values = torch.bmm(values, scores.transpose(1,2))
        attn_values = attn_values.view(batch_size, channels, height, width)
        
        return attn_values

class ChannelSelfAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input x has shape (batch_size, channels, height, width)
        
        # Compute global average pool
        x_pool = self.avg_pool(x)
        
        # Compute attention weights
        weights = self.conv1(x_pool)
        weights = torch.relu(weights)
        weights = self.conv2(weights)
        weights = self.sigmoid(weights)
        
        # Apply attention to input features
        x_att = x * weights
        
        return x_att