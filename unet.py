import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# class UNet(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3):
#         super(UNet, self).__init__()

#         def conv_block(in_c, out_c):
#             return nn.Sequential(
#                 nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
#                 nn.ReLU(inplace=True),
#             )

#         self.encoder1 = conv_block(in_channels, 64)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.encoder2 = conv_block(64, 128)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.encoder3 = conv_block(128, 256)
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.encoder4 = conv_block(256, 512)
#         self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.bottleneck = conv_block(512, 1024)

#         self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
#         self.decoder4 = conv_block(1024, 512)

#         self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
#         self.decoder3 = conv_block(512, 256)

#         self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.decoder2 = conv_block(256, 128)

#         self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.decoder1 = conv_block(128, 64)

#         self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

#     def forward(self, x):
#         e1 = self.encoder1(x)
#         p1 = self.pool1(e1)

#         e2 = self.encoder2(p1)
#         p2 = self.pool2(e2)

#         e3 = self.encoder3(p2)
#         p3 = self.pool3(e3)

#         e4 = self.encoder4(p3)
#         p4 = self.pool4(e4)

#         b = self.bottleneck(p4)

#         u4 = self.upconv4(b)
#         d4 = self.decoder4(torch.cat((u4, e4), dim=1))

#         u3 = self.upconv3(d4)
#         d3 = self.decoder3(torch.cat((u3, e3), dim=1))

#         u2 = self.upconv2(d3)
#         d2 = self.decoder2(torch.cat((u2, e2), dim=1))

#         u1 = self.upconv1(d2)
#         d1 = self.decoder1(torch.cat((u1, e1), dim=1))

#         output = self.final_conv(d1)
#         return output

# # Instantiate the model
# model = UNet(in_channels=3, out_channels=3)
# print(model)





class UNet(nn.Module):
    def __init__(self, encoder_name="resnet34", in_channels=3, out_channels=1):
        """
        Custom UNet with a ResNet34 encoder.

        Args:
            encoder_name (str): Name of the encoder. Currently, only 'resnet34' is supported.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output classes (1 for binary, N for multi-class segmentation).
        """
        super(UNet, self).__init__()

        if encoder_name == "resnet34":
            encoder = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        else:
            raise NotImplementedError("Only 'resnet34' is currently supported.")

        # Encoder parts (using ResNet34's layers)
        self.initial = nn.Sequential(
            encoder.conv1,  
            encoder.bn1,
            encoder.relu
        )
        self.maxpool = encoder.maxpool  
        self.layer1 = encoder.layer1  
        self.layer2 = encoder.layer2  
        self.layer3 = encoder.layer3  
        self.layer4 = encoder.layer4  

        # Decoder: each upsampling step concatenates with corresponding encoder feature map.
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(512, 256)  

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 128)  

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64)   

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)   

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)  

    def conv_block(self, in_channels, out_channels):
        """Helper function to create a convolutional block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        x0 = self.initial(x)         
        x1 = self.maxpool(x0)        
        x1 = self.layer1(x1)         
        x2 = self.layer2(x1)         
        x3 = self.layer3(x2)         
        x4 = self.layer4(x3)         

        # Decoder
        d4 = self.up4(x4)            
        d4 = torch.cat([d4, x3], dim=1)  
        d4 = self.dec4(d4)           

        d3 = self.up3(d4)            
        d3 = torch.cat([d3, x2], dim=1)  
        d3 = self.dec3(d3)           

        d2 = self.up2(d3)            
        d2 = torch.cat([d2, x1], dim=1)  
        d2 = self.dec2(d2)           

        d1 = self.up1(d2)            
        d1 = torch.cat([d1, x0], dim=1)  
        d1 = self.dec1(d1)           

        out = self.final_conv(d1)   

        # Apply activation based on task type
        if out.shape[1] == 1:  # Binary segmentation
            out = torch.sigmoid(out)  # Output is probability map (0 to 1)
        else:  # Multi-class segmentation
            out = torch.softmax(out, dim=1)  # Output is class probability distribution

        return out