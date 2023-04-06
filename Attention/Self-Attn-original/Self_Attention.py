# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 01:57:50 2023

@author: User
"""

import torch
import torch.nn as nn
'''
This implementation defines a SelfAttention module that takes an input feature map of size (batch_size, channels, height, width) 
and computes a self-attention map using three convolutional layers.
 The query_conv, key_conv, and value_conv convolutional layers project the input feature maps into the query, key, and value spaces, respectively.
 The softmax function is applied to the energy tensor to compute the attention weights,
 which are then used to weight the value tensor.
 The output is computed by applying a convolutional layer to the weighted value tensor and adding it to the original input tensor.

The query_channels, key_channels, and value_channels parameters control the number of output channels for the query,
 key, and value convolutions, respectively.
 If these parameters are not specified,
 the default values are set to one-eighth of the input channel dimension for the query and key convolutions,
 and the full input channel dimension for the value convolution.
'''
class SelfAttention(nn.Module):
    def __init__(self, in_channels, query_channels=None, key_channels=None, value_channels=None):
        super(SelfAttention, self).__init__()
        self.query_channels = query_channels if query_channels is not None else in_channels // 8
        self.key_channels = key_channels if key_channels is not None else in_channels // 8
        self.value_channels = value_channels if value_channels is not None else in_channels
        self.query_conv = nn.Conv2d(in_channels, self.query_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, self.key_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, self.value_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(self.value_channels, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Compute query, key, and value
        query = self.query_conv(x).view(batch_size, self.query_channels, -1)
        key = self.key_conv(x).view(batch_size, self.key_channels, -1)
        value = self.value_conv(x).view(batch_size, self.value_channels, -1)

        # Compute attention weights
        energy = torch.bmm(query.permute(0, 2, 1), key)
        attention = self.softmax(energy)

        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, self.value_channels, height, width)

        # Apply output convolution
        out = self.out_conv(out)
        out += x

        return out
