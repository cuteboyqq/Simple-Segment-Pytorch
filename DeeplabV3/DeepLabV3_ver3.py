# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 00:47:34 2023

@author: User
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.models as models
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(ASPP, self).__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, kernel_size, padding, dilation)
        self.conv2 = ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0, dilation=1)
        self.conv3 = ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, groups=4)
        self.conv4 = ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, groups=8)
        self.conv5 = ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, groups=16)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0, dilation=1)
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        out5 = self.conv5(x)
        out6 = self.pool(x)
        out6 = F.interpolate(out6, size=(out1.size(2), out1.size(3)), mode="bilinear", align_corners=True)

        out = torch.cat([out1, out2, out3, out4, out5, out6], dim=1)
        return out


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()

        # ASPP modules with different dilation rates
        self.aspp1 = ASPP(2048, 256, 1, 1, 0)
        self.aspp2 = ASPP(2048, 256, 3, 1, 6)
        self.aspp3 = ASPP(2048, 256, 3, 1, 12)
        self.aspp4 = ASPP(2048, 256, 3, 1, 18)

        # Global average pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvBNReLU(2048, 256, 1, bias=False)
        )

        # Concatenate ASPP outputs and global average pooling output
        self.conv1 = ConvBNReLU(1280, 256, 1, bias=False)
        self.conv2 = ConvBNReLU(256, 256, 3, padding=1, bias=False)

        # Final convolution to produce the segmentation map
        self.conv3 = nn.Conv2d(256, num_classes, 1)

    def forward(self, x, low_level_features):
        # Upsample low-level features to the same size as x
        low_level_features = F.interpolate(low_level_features, size=x.shape[2:], mode='bilinear', align_corners=True)

        # Apply ASPP modules
        aspp1 = self.aspp1(x)
        aspp2 = self.aspp2(x)
        aspp3 = self.aspp3(x)
        aspp4 = self.aspp4(x)
        global_avg_pool = self.global_avg_pool(x)
        global_avg_pool = F.interpolate(global_avg_pool, size=aspp1.shape[2:], mode='bilinear', align_corners=True)

        # Concatenate ASPP outputs and global average pooling output
        concat = torch.cat((aspp1, aspp2, aspp3, aspp4, global_avg_pool), dim=1)
        concat = self.conv1(concat)
        concat = F.interpolate(concat, size=low_level_features.shape[2:], mode='bilinear', align_corners=True)
        concat = torch.cat((concat, low_level_features), dim=1)
        concat = self.conv2(concat)

        # Final convolution to produce the segmentation map
        output = self.conv3(concat)
        output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=True)

        return output


class DeepLabV3(nn.Module):
    def __init__(self, num_classes=21, output_stride=16, input_size=(512, 512)):
        super(DeepLabV3, self).__init__()
        self.output_stride = output_stride
        self.input_size = input_size
        
        # Define the backbone network
        self.backbone = models.resnet50(pretrained=True)
        
        # Remove the classification head
        del self.backbone.avgpool
        del self.backbone.fc
        
        # Modify the stride of the last convolutional block
        if self.output_stride == 8:
            self.backbone.layer4[0].conv2.stride = (1, 1)
            self.backbone.layer4[0].downsample[0].stride = (1, 1)
        
        # Define the ASPP module
        self.aspp = ASPP(in_channels=2048, output_stride=self.output_stride)
        
        # Define the decoder network
        self.decoder = Decoder(num_classes=num_classes, input_size=self.input_size)
        
        # Define the final convolutional layer
        self.final_conv = nn.Conv2d(256, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Backbone network
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # ASPP module
        x = self.aspp(x)

        # Decoder network
        x = self.decoder(x)

        # Final convolutional layer
        x = self.final_conv(x)
        
        # Upsample the output to the original size
        x = F.interpolate(x, size=self.input_size, mode='bilinear', align_corners=True)

        return x

