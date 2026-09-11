# Enhanced DynGAN with Dynamic Parameter Tuning and Global Positional Encoding

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.unet import ConvDDynamic, CoordConv2d

class CalibrationBlock(nn.Module):
    def __init__(self, in_channels, img_size, hidden_dim=256):
        super(CalibrationBlock, self).__init__()

        # total input features per sample: C * H * W
        in_features = in_channels * img_size * img_size  # now supports 128×128

        # MLP: C*H*W → hidden_dim → H*W (spatial mask only)
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, img_size * img_size)

    def forward(self, x):
        B, C, H, W = x.shape

        # Flatten full 128×128 patch
        x_flat = x.reshape(B, C * H * W)

        # Pass through FC layers sized for 128×128 = 16384
        v = torch.sigmoid(self.fc2(torch.sigmoid(self.fc1(x_flat))))

        # Reshape back to 128×128
        v = v.reshape(B, 1, H, W)

        # Apply calibration mask
        return x * v


class DynamicUNetBlock(nn.Module):
    """
    Enhanced U-Net with Dynamic Convolutions and Global Positional Encoding
    """
    def __init__(self, in_ch=1, base_ch=32, use_coord_conv=True):
        super().__init__()
        self.use_coord_conv = use_coord_conv
        
        self.enc1 = ConvDDynamic(in_ch, base_ch, norm='bn', first=True, activation='leaky')
        self.enc2 = ConvDDynamic(base_ch, base_ch*2, norm='bn', activation='leaky')
        self.enc3 = ConvDDynamic(base_ch*2, base_ch*4, norm='bn', activation='leaky')
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_ch*4, base_ch*8, 4, 2, 1),
            nn.BatchNorm2d(base_ch*8),
            nn.LeakyReLU(0.2)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(base_ch*8, base_ch*16, 4, 2, 1),
            nn.BatchNorm2d(base_ch*16),
            nn.LeakyReLU(0.2)
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*16, base_ch*8, 4, 2, 1),
            nn.BatchNorm2d(base_ch*8),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*16, base_ch*4, 4, 2, 1),
            nn.BatchNorm2d(base_ch*4),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*8, base_ch*2, 4, 2, 1),
            nn.BatchNorm2d(base_ch*2),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*4, base_ch, 4, 2, 1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        if self.use_coord_conv:
            self.coord_conv = CoordConv2d(2*base_ch, base_ch, kernel_size=1, padding=0, bias=False)
        
        self.out = nn.Sequential(
            nn.Conv2d(base_ch, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        d1 = self.dec1(e5)
        d1 = torch.cat([d1, e4], dim=1)

        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e3], dim=1)

        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e2], dim=1)

        d4 = self.dec4(d3)
        d4 = torch.cat([d4, e1], dim=1)

        if self.use_coord_conv:
            d4 = self.coord_conv(d4)

        return self.out(d4)


class DynGANGeneratorDynamic(nn.Module):
    """
    DynGAN with Dynamic Parameters and Global Positional Encoding
    """

    def __init__(self, w=128, base_ch=32, use_coord_conv=True):
        super(DynGANGeneratorDynamic, self).__init__()

        in_channels = 1  # dynamic GAN uses FLAIR-only input
        self.calib = CalibrationBlock(in_channels=in_channels, img_size=w)

        self.dynamic_unet = DynamicUNetBlock(
            in_ch=in_channels,
            base_ch=base_ch,
            use_coord_conv=use_coord_conv
        )

    def forward(self, x):
        x = self.calib(x)
        y = self.dynamic_unet(x)
        return y
