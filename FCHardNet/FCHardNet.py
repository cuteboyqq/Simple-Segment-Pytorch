# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 01:29:34 2023

@author: User
"""

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)
'''
To modify FCHarDNet for semantic segmentation, 
we can add a decoder to the end of the network to upsample the feature maps and generate the final segmentation output. 
Here is an example implementation of FCHarDNet for semantic segmentation in PyTorch:
'''
        
class FCHarDNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.encoder = nn.Sequential(
            ConvBlock(3, 16, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Sequential(*[ResBlock(16, dilation=1) for _ in range(4)]),
            nn.MaxPool2d(2, 2),
            nn.Sequential(*[ResBlock(32, dilation=1) for _ in range(4)]),
            nn.MaxPool2d(2, 2),
            nn.Sequential(*[ResBlock(64, dilation=1) for _ in range(4)]),
            nn.MaxPool2d(2, 2),
            nn.Sequential(*[ResBlock(128, dilation=1) for _ in range(4)]),
            nn.MaxPool2d(2, 2),
            nn.Sequential(*[ResBlock(256, dilation=1) for _ in range(4)]),
            nn.MaxPool2d(2, 2),
            nn.Sequential(*[ResBlock(512, dilation=1) for _ in range(4)]),
        )
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(512, 256, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(256, 128, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(128, 64, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(64, 32, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(32, 16, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        
        self.classifier = nn.Conv2d(16, num_classes, 1)
        
    def forward(self, x):
        features = self.encoder(x)
        decoded = self.decoder(features)
        logits = self.classifier(decoded)
        return logits

'''
The ResBlock is a basic building block for FCHarDNet, 
and consists of two convolutional blocks with residual connections between them. 
The dilation parameter controls the spacing between the kernel elements,
 allowing for larger receptive fields without increasing the number of parameters.
'''
    
class ResBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        
        self.conv1 = ConvBlock(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = ConvBlock(channels, channels, 3, padding=dilation, dilation=dilation)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += x
        return out

