# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 02:38:19 2023

@author: User
"""

import torch
import torch.nn as nn

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.block = nn.Sequential(ConvBNReLU(in_channels, out_channels), ConvBNReLU(out_channels, out_channels))

    def forward(self, x):
        x = self.block(x)
        x_down = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        return x_down, x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.conv_up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(ConvBNReLU(out_channels * 2, out_channels), ConvBNReLU(out_channels, out_channels))

    def forward(self, x, x_down):
        x_up = self.conv_up(x)
        x_cat = torch.cat([x_down, x_up], dim=1)
        x = self.conv(x_cat)
        return x

class FCHardNet(nn.Module):
    def __init__(self, num_classes=10):
        super(FCHardNet, self).__init__()
        self.conv1 = ConvBNReLU(3, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.up5 = Up(64, 32)
        self.conv_last = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2, x_down1 = self.down1(x1)
        x3, x_down2 = self.down2(x2)
        x4, x_down3 = self.down3(x3)
        x5, x_down4 = self.down4(x4)
        x6, x_down5 = self.down5(x5)
        x = self.up1(x6, x_down5)
        x = self.up2(x, x_down4)
        x = self.up3(x, x_down3)
        x = self.up4(x, x_down2)
        x = self.up5(x, x_down1)
        x = self.conv_last(x)
        return x
