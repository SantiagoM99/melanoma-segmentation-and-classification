import torch
import torch.nn as nn


# Define the conv_block (2 convolution layers + ReLU)
class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# Define the up_conv block (transposed convolution for upsampling)
class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Encoding path
        self.encoder1 = nn.Sequential(
            ConvBlock(ch_in=in_channels, ch_out=32)  # (3, 32)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = nn.Sequential(ConvBlock(ch_in=32, ch_out=64))  # (32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = nn.Sequential(ConvBlock(ch_in=64, ch_out=128))  # (64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = nn.Sequential(ConvBlock(ch_in=128, ch_out=256))  # (128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(ConvBlock(ch_in=256, ch_out=512))  # (256, 512)

        # Decoding path
        self.upconv4 = UpConv(ch_in=512, ch_out=256)
        self.decoder4 = nn.Sequential(ConvBlock(ch_in=512, ch_out=256))  # (512, 256)

        self.upconv3 = UpConv(ch_in=256, ch_out=128)
        self.decoder3 = nn.Sequential(ConvBlock(ch_in=256, ch_out=128))  # (256, 128)

        self.upconv2 = UpConv(ch_in=128, ch_out=64)
        self.decoder2 = nn.Sequential(ConvBlock(ch_in=128, ch_out=64))  # (128, 64)

        self.upconv1 = UpConv(ch_in=64, ch_out=32)
        self.decoder1 = nn.Sequential(ConvBlock(ch_in=64, ch_out=32))  # (64, 32)

        # Final 1x1 conv to get the output
        self.Conv_1x1 = nn.Conv2d(32, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoding path
        x1 = self.encoder1(x)  # (B, 32, H, W)
        x2 = self.pool1(x1)  # (B, 32, H//2, W//2)
        x2 = self.encoder2(x2)  # (B, 64, H//2, W//2)

        x3 = self.pool2(x2)  # (B, 64, H//4, W//4)
        x3 = self.encoder3(x3)  # (B, 128, H//4, W//4)

        x4 = self.pool3(x3)  # (B, 128, H//8, W//8)
        x4 = self.encoder4(x4)  # (B, 256, H//8, W//8)

        x5 = self.pool4(x4)  # (B, 256, H//16, W//16)
        x5 = self.bottleneck(x5)  # (B, 512, H//16, W//16)

        # Decoding path with skip connections
        d4 = self.upconv4(x5)  # (B, 256, H//8, W//8)
        d4 = torch.cat((x4, d4), dim=1)  # Concatenate with encoder4 output
        d4 = self.decoder4(d4)  # (B, 256, H//8, W//8)

        d3 = self.upconv3(d4)  # (B, 128, H//4, W//4)
        d3 = torch.cat((x3, d3), dim=1)  # Concatenate with encoder3 output
        d3 = self.decoder3(d3)  # (B, 128, H//4, W//4)

        d2 = self.upconv2(d3)  # (B, 64, H//2, W//2)
        d2 = torch.cat((x2, d2), dim=1)  # Concatenate with encoder2 output
        d2 = self.decoder2(d2)  # (B, 64, H//2, W//2)

        d1 = self.upconv1(d2)  # (B, 32, H, W)
        d1 = torch.cat((x1, d1), dim=1)  # Concatenate with encoder1 output
        d1 = self.decoder1(d1)  # (B, 32, H, W)

        # Final 1x1 conv to get the output
        output = self.Conv_1x1(d1)  # (B, out_channels, H, W)

        return output
