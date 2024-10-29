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
        self.rcnn = nn.Sequential(
            RecurrentBlock(ch_out, t=t),
            RecurrentBlock(ch_out, t=t)
        )
        self.conv1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

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

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.r2conv1 = ResidualRecurrentBlock(ch_in=in_channels, ch_out=32, t=t)

        self.r2conv2 = ResidualRecurrentBlock(ch_in=32, ch_out=64, t=t)

        self.r2conv3 = ResidualRecurrentBlock(ch_in=64, ch_out=128, t=t)

        self.r2conv4 = ResidualRecurrentBlock(ch_in=128, ch_out=256, t=t)

        self.r2conv5 = ResidualRecurrentBlock(ch_in=256, ch_out=512, t=t)

        self.upconv1 = UpConv(ch_in=512, ch_out=256)
        self.r2decod1 = ResidualRecurrentBlock(ch_in=512, ch_out=256, t=t)

        self.upconv2 = UpConv(ch_in=256, ch_out=128)
        self.r2decod2 = ResidualRecurrentBlock(ch_in=256, ch_out=128, t=t)

        self.upconv3 = UpConv(ch_in=128, ch_out=64)
        self.r2decod3 = ResidualRecurrentBlock(ch_in=128, ch_out=64, t=t)

        self.upconv4 = UpConv(ch_in=64, ch_out=32)
        self.r2decod4 = ResidualRecurrentBlock(ch_in=64, ch_out=32, t=t)

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
        x1 = self.r2conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.r2conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.r2conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.r2conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.r2conv5(x5)

        # Decoding path
        d5 = self.upconv1(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.r2decod1(d5)

        d4 = self.upconv2(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.r2decod2(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.r2decod3(d3)

        d2 = self.upconv4(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.r2decod4(d2)

        out = self.Conv_1x1(d2)

        return out