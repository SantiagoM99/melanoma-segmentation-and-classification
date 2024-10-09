import torch
import torch.nn as nn
from models.unet import Encoder, Decoder


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Conv2d(
            in_channels, inter_channels, kernel_size=1, stride=1, padding=0
        )
        self.W_x = nn.Conv2d(
            out_channels, inter_channels, kernel_size=1, stride=1, padding=0
        )
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = torch.relu(g1 + x1)
        psi = self.psi(psi)
        psi = torch.sigmoid(psi)
        return x * psi


class AttUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttUNet, self).__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(out_channels)
        self.attention = AttentionBlock(1024, 512, 512)

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        x3 = self.attention(x1, x2)
        return x3
