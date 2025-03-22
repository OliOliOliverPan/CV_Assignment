import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        # Encoding Path
        self.encoder1 = conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = conv_block(128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = conv_block(256, 512)

        # Decoding Path
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = conv_block(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = conv_block(256, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = conv_block(128, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = conv_block(64, 32)

        # Final Output Layer
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()  # Binary segmentation

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)

        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)

        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)

        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder
        u4 = self.upconv4(b)
        d4 = self.decoder4(torch.cat((u4, e4), dim=1))

        u3 = self.upconv3(d4)
        d3 = self.decoder3(torch.cat((u3, e3), dim=1))

        u2 = self.upconv2(d3)
        d2 = self.decoder2(torch.cat((u2, e2), dim=1))

        u1 = self.upconv1(d2)
        d1 = self.decoder1(torch.cat((u1, e1), dim=1))

        output = self.final_conv(d1)
        return output # Raw logits output
        # return self.activation(output)  # Apply sigmoid for binary segmentation