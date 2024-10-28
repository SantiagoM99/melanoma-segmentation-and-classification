import torch
import torch.nn as nn
from melanoma_segmentation.models.unet import UpConv

class RecurrentBlock(nn.Module):
    """
    Recurrent block used in the R2U-Net model to apply multiple convolutions on the input feature map.

    Parameters
    ----------
    ch_out : int
        Number of output channels.
    t : int, optional
        Number of recurrent steps (default is 2).

    Attributes
    ----------
    conv : nn.Sequential
        A sequence of convolution, batch normalization, and ReLU activation layers.

    Methods
    -------
    forward(x)
        Performs the forward pass of the recurrent block.
    """
    
    def __init__(self, ch_out, t=2):
        super(RecurrentBlock, self).__init__()
        self.t = t
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass for the Recurrent Block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor after recurrent convolution operations.
        """
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            else:
                x1 = self.conv(x + x1)
        return x1


class ResidualRecurrentBlock(nn.Module):
    """
    Residual block with recurrent convolutions for the R2U-Net model.

    Parameters
    ----------
    ch_in : int
        Number of input channels.
    ch_out : int
        Number of output channels.
    t : int, optional
        Number of recurrent steps (default is 2).

    Attributes
    ----------
    conv1x1 : nn.Conv2d
        A 1x1 convolutional layer to match the dimensions between input and output channels.
    rcnn : RecurrentBlock
        A recurrent block for repeated convolution operations.

    Methods
    -------
    forward(x)
        Performs the forward pass of the residual recurrent block.
    """
    
    def __init__(self, ch_in, ch_out, t=2):
        super(ResidualRecurrentBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
        self.rcnn = RecurrentBlock(ch_out, t=t)

    def forward(self, x):
        """
        Forward pass for the Residual Recurrent Block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor with residual connections.
        """
        x = self.conv1x1(x)
        x1 = self.rcnn(x)
        return x + x1  # Residual connection


class R2UNet(nn.Module):
    """
    R2U-Net architecture for image segmentation tasks.

    This model is an extension of U-Net using residual recurrent blocks to improve feature learning
    by adding recurrent convolutions and residual connections.

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels. Default is 3.
    out_channels : int, optional
        Number of output channels. Default is 1.
    t : int, optional
        Number of recurrent steps in each block. Default is 2.

    Attributes
    ----------
    encoder1, encoder2, encoder3, encoder4 : nn.Sequential
        Encoder blocks consisting of residual recurrent blocks.
    pool1, pool2, pool3, pool4 : nn.MaxPool2d
        Pooling layers for downsampling the feature maps.
    bottleneck : nn.Sequential
        Bottleneck block connecting the encoder and decoder paths.
    upconv1, upconv2, upconv3, upconv4 : UpConv
        Upsampling layers for increasing the feature map size in the decoder path.
    decoder1, decoder2, decoder3, decoder4 : nn.Sequential
        Decoder blocks consisting of residual recurrent blocks.
    Conv_1x1 : nn.Conv2d
        Final 1x1 convolutional layer for producing the output segmentation map.

    Methods
    -------
    forward(x)
        Performs the forward pass of the R2U-Net model.
    """
    
    def __init__(self, in_channels=3, out_channels=1, t=2):
        super(R2UNet, self).__init__()

        # Encoding path
        self.encoder1 = nn.Sequential(ResidualRecurrentBlock(ch_in=in_channels, ch_out=32, t=t))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = nn.Sequential(ResidualRecurrentBlock(ch_in=32, ch_out=64, t=t))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = nn.Sequential(ResidualRecurrentBlock(ch_in=64, ch_out=128, t=t))
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = nn.Sequential(ResidualRecurrentBlock(ch_in=128, ch_out=256, t=t))
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(ResidualRecurrentBlock(ch_in=256, ch_out=512, t=t))

        # Decoding path
        self.upconv4 = UpConv(ch_in=512, ch_out=256)
        self.decoder4 = nn.Sequential(ResidualRecurrentBlock(ch_in=512, ch_out=256, t=t))

        self.upconv3 = UpConv(ch_in=256, ch_out=128)
        self.decoder3 = nn.Sequential(ResidualRecurrentBlock(ch_in=256, ch_out=128, t=t))

        self.upconv2 = UpConv(ch_in=128, ch_out=64)
        self.decoder2 = nn.Sequential(ResidualRecurrentBlock(ch_in=128, ch_out=64, t=t))

        self.upconv1 = UpConv(ch_in=64, ch_out=32)
        self.decoder1 = nn.Sequential(ResidualRecurrentBlock(ch_in=64, ch_out=32, t=t))

        # Final 1x1 conv to get the output
        self.Conv_1x1 = nn.Conv2d(32, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Forward pass for the R2U-Net model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, H, W) where:
            - N is the batch size
            - C is the number of input channels
            - H is the height of the input image
            - W is the width of the input image

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, out_channels, H, W).
        """
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
