from melanoma_segmentation.models.unet import ConvBlock, UpConv
from melanoma_segmentation.models.attention_unet import AttentionBlock
from melanoma_segmentation.models.residual_recurrent_unet import RecurrentBlock, ResidualRecurrentBlock

import torch
from torch import nn
from torch.nn import functional as F

class R2AttUNet(nn.Module):
    """
    R2AttUNet model for image segmentation tasks.

    This model combines Residual Recurrent Convolutions with Attention mechanisms in a U-Net-like architecture.
    It consists of an encoding path to extract features and a decoding path with skip connections for 
    spatial information recovery.

    Parameters
    ----------
    ch_in : int
        Number of input channels.
    ch_out : int
        Number of output channels.
    t : int, optional
        Number of recurrent steps in each block (default is 2).
    stride : int, optional
        Stride value for the MaxPool2d layers (default is 2).

    Attributes
    ----------
    Maxpool : nn.MaxPool2d
        Max pooling layer for downsampling the feature maps.
    Up_Sample : nn.Upsample
        Upsampling layer to increase feature map resolution in the decoding path.
    r2conv1, r2conv2, r2conv3, r2conv4, r2conv5 : ResidualRecurrentBlock
        Residual recurrent convolutional blocks in the encoding path.
    upconv1, upconv2, upconv3, upconv4 : UpConv
        Upsampling layers in the decoding path.
    attention1, attention2, attention3, attention4 : AttentionBlock
        Attention blocks to focus on important features during decoding.
    r2decod1, r2decod2, r2decod3, r2decod4 : RecurrentBlock
        Recurrent convolutional blocks in the decoding path.
    final_conv : nn.Conv2d
        Final 1x1 convolutional layer to produce the output segmentation map.

    Methods
    -------
    forward(x)
        Performs the forward pass of the R2AttUNet model.
    """

    def __init__(self, ch_in=3, ch_out=1, t=2, stride=2):
        super(R2AttUNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=stride)
        self.Up_Sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.r2conv1 = ResidualRecurrentBlock(ch_in=ch_in, ch_out=32, t=t)
        self.r2conv2 = ResidualRecurrentBlock(ch_in=32, ch_out=64, t=t)
        self.r2conv3 = ResidualRecurrentBlock(ch_in=64, ch_out=128, t=t)
        self.r2conv4 = ResidualRecurrentBlock(ch_in=128, ch_out=256, t=t)
        self.r2conv5 = ResidualRecurrentBlock(ch_in=256, ch_out=512, t=t)


        self.upconv1 = UpConv(512, 256)
        self.attention1 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.r2decod1 = ResidualRecurrentBlock(ch_in=512, ch_out=256, t=t)

        self.upconv2 = UpConv(256, 128)
        self.attention2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.r2decod2 = ResidualRecurrentBlock(ch_in=256, ch_out=128, t=t)

        self.upconv3 = UpConv(128, 64)
        self.attention3 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.r2decod3 = ResidualRecurrentBlock(ch_in=128, ch_out=64, t=t)

        self.upconv4 = UpConv(64, 32)
        self.attention4 = AttentionBlock(F_g=32, F_l=32, F_int=16)
        self.r2decod4 = ResidualRecurrentBlock(ch_in=64, ch_out=32, t=t)

        self.final_conv = nn.Conv2d(32, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Forward pass for the R2AttUNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, H, W) where:
            - N is the batch size.
            - C is the number of input channels.
            - H is the height of the input image.
            - W is the width of the input image.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, ch_out, H, W).
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
        d4 = self.upconv1(x5)
        x4 = self.attention1(g=d4, x=x4)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.r2decod1(d4)

        d3 = self.upconv2(d4)
        x3 = self.attention2(g=d3, x=x3)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.r2decod2(d3)

        d2 = self.upconv3(d3)
        x2 = self.attention3(g=d2, x=x2)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.r2decod3(d2)

        d1 = self.upconv4(d2)
        x1 = self.attention4(g=d1, x=x1)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.r2decod4(d1)

        out = self.final_conv(d1)

        return out
