
# models/degan_discriminator.py

import torch
import torch.nn as nn

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=2, base_ch=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(base_ch, base_ch*2, 4, 2, 1), nn.BatchNorm2d(base_ch*2), nn.LeakyReLU(0.2),
            nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1), nn.BatchNorm2d(base_ch*4), nn.LeakyReLU(0.2),
            nn.Conv2d(base_ch*4, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.model(x)

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, in_ch=2, base_ch=64, num_scales=2):
        super().__init__()
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList([
            PatchDiscriminator(in_ch, base_ch) for _ in range(num_scales)
        ])
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)
        outputs = []
        for i in range(self.num_scales):
            outputs.append(self.discriminators[i](inp))
            inp = self.downsample(inp)
        return outputs
