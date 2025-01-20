"""
25.12.24

Claude 3.5 Sonnet
"""

import torch
import torch.nn as nn
from torch import Tensor as _T
import torch.nn.functional as F

from functools import partial

from torch import vmap


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: _T) -> _T:
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        image_size: int,
        num_channels: int,
        vector_dim: int,
        base_channels: int = 64,
    ):
        """
        UNet architecture with customizable channel scaling and vector input support.

        Args:
            image_size: Size of the input images (assumes square images)
            num_channels: Number of input and output channels
            base_channels: Number of channels in first encoder block (others scale from this)
            vector_dim: Dimension of additional vector input to be concatenated after bridge
        """
        super().__init__()

        self.image_shape = (num_channels, image_size, image_size)

        # Calculate channel sizes based on base_channels
        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2
        c5 = c4 * 2  # bridge channels

        # Encoder (downsampling)
        self.enc1 = DoubleConv(num_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)

        # Bridge
        self.bridge = DoubleConv(c4, c5)

        # Calculate sizes for vector integration
        self.vector_dim = vector_dim
        bridge_size = image_size // 16  # After 4 downsamplings
        self.bridge_channels = c5
        self.bridge_spatial_size = bridge_size

        # Vector processing if needed
        self.vector_proj = nn.Linear(vector_dim, bridge_size * bridge_size)
        c5 = c5 + 1  # Add 1 channel for vector information

        # Decoder (upsampling)
        self.up1 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c4 * 2, c4)  # *2 due to concatenation

        self.up2 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c3 * 2, c3)

        self.up3 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c2 * 2, c2)

        self.up4 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c1 * 2, c1)

        # Final convolution
        self.final_conv = nn.Conv2d(c1, num_channels, kernel_size=1)

        self.float()

        # Input size check
        if image_size % 16 != 0:
            raise ValueError(
                "Image size must be divisible by 16 for proper UNet operation"
            )

    def forward(self, x: _T, vector_input: _T) -> _T:
        """
        Forward pass through the UNet.

        Args:
            x: Input tensor of shape [batch_size, T, num_channels, image_size, image_size]
            vector_input: vector input of shape [batch_size, T, vector_dim]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, T, num_channels, image_size, image_size]
        """

        batch_size, timesteps = x.shape[:2]
        x = x.reshape(batch_size * timesteps, *self.image_shape).float()
        import pdb

        pdb.set_trace(header="make sure whatever rework done here actually works!")

        # x == x.reshape(batch_size * timesteps, *self.image_shape).reshape(batch_size, timesteps, *self.image_shape)

        # Encoder
        enc1 = self.enc1(x)
        x = F.max_pool2d(enc1, 2)

        enc2 = self.enc2(x)
        x = F.max_pool2d(enc2, 2)

        enc3 = self.enc3(x)
        x = F.max_pool2d(enc3, 2)

        enc4 = self.enc4(x)
        x = F.max_pool2d(enc4, 2)

        # Bridge
        x = self.bridge(x)  # Shape: [batch_size, c5, bridge_size, bridge_size]

        # Project vector to spatial dimension
        v = self.vector_proj(vector_input)  # [batch_size, bridge_size * bridge_size]
        v = v.view(-1, 1, self.bridge_spatial_size, self.bridge_spatial_size)
        x = torch.cat([x, v], dim=1)

        # Decoder
        x = self.up1(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec2(x)

        x = self.up3(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec3(x)

        x = self.up4(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec4(x)

        # Final convolution
        x = self.final_conv(x)

        x = x.reshape(batch_size, timesteps, *self.image_shape)

        return x

    def count_parameters(self):
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
