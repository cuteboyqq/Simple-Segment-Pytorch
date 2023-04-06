# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 01:12:39 2023

@author: User
"""
import torch
import torch.nn as nn
from Self_Attention import SpatialSelfAttention, ChannelSelfAttention
import torch.nn.functional as F
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x
    
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )
        self.spatial_attention = SpatialSelfAttention(out_channels)
        self.channel_attention = ChannelSelfAttention(out_channels)
        
    def forward(self, x):
        x = self.mpconv(x)
        x = self.spatial_attention(x)
        x = self.channel_attention(x)
        return x
    
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # If using bilinear interpolation, use the ConvTranspose2d layer to perform upsampling
        if bilinear:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv = DoubleConv(in_channels, out_channels)
        self.spatial_attention = SpatialSelfAttention(out_channels)
        self.channel_attention = ChannelSelfAttention(out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Adjust the size of the upsampled feature maps if the sizes don't match
        diffH = x2.size()[2] - x1.size()[2]
        diffW = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2])
        
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.spatial_attention(x)
        x = self.channel_attention(x)
        return x



'''
In the forward method, we first apply the double convolutional layers to the input image to get the output feature maps of the first contracting block.
 Then we apply the subsequent contracting blocks using the Down module. 
 At the bottom layer, we apply the DoubleConv module again. 
 Then we apply the expanding path using the Up module, 
 concatenating the upsampled feature maps with the corresponding feature maps from the contracting path, 
 and applying the double convolutional layers. 
 Finally, we apply the output convolutional layer and softmax activation to get the predicted class probabilities,
 and then apply the spatial self-attention and channel self-attention modules to the output feature maps.
 '''
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.bridge = DoubleConv(512, 1024)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        self.spatial_attention = SpatialSelfAttention(64)
        self.channel_attention = ChannelSelfAttention(64)
        
    def forward(self, x):
        # Contracting path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bottom layer
        x5 = self.bridge(x4)

        # Expanding path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = self.softmax(x)
        x = self.spatial_attention(x)
        x = self.channel_attention(x)

        return x
