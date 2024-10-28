import torch
import torch.nn as nn
from models.unet import ConvBlock, UpConv


class AttentionBlock(nn.Module):
    """
    Attention Block for the Attention U-Net model.

    This block performs attention-based gating, which emphasizes the most relevant features 
    from the skip connections, improving segmentation accuracy.

    Parameters
    ----------
    F_g : int
        Number of channels in the gating signal (decoder feature map).
    F_l : int
        Number of channels in the skip connection signal.
    F_int : int
        Number of intermediate channels used in the attention mechanism.

    Attributes
    ----------
    W_g : nn.Sequential
        Convolution and Batch Normalization applied to the gating signal.
    W_x : nn.Sequential
        Convolution and Batch Normalization applied to the skip connection.
    psi : nn.Sequential
        Final 1x1 convolution, followed by a sigmoid activation for producing the attention weights.
    relu : nn.ReLU
        ReLU activation applied to the combined feature map.

    Methods
    -------
    forward(g, x)
        Performs the forward pass of the attention block.
    """
    
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """
        Forward pass for the Attention Block.

        Parameters
        ----------
        g : torch.Tensor
            Gating signal (decoder feature map).
        x : torch.Tensor
            Skip connection feature map.

        Returns
        -------
        torch.Tensor
            Attention-weighted feature map.
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttUNet(nn.Module):
    """
    Attention U-Net architecture for image segmentation tasks.

    This model uses an encoder-decoder architecture with attention gates in the skip connections to focus on
    relevant features, improving segmentation results.

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels. Default is 3.
    out_channels : int, optional
        Number of output channels. Default is 1.

    Attributes
    ----------
    encoder1, encoder2, encoder3, encoder4 : ConvBlock
        Encoder convolutional blocks for extracting features at different scales.
    pool1, pool2, pool3, pool4 : nn.MaxPool2d
        Pooling layers for downsampling the feature maps.
    bottleneck : ConvBlock
        Bottleneck convolutional block connecting the encoder and decoder paths.
    upconv1, upconv2, upconv3, upconv4 : UpConv
        Upsampling layers for increasing the feature map size in the decoder path.
    attention1, attention2, attention3, attention4 : AttentionBlock
        Attention blocks applied to the skip connections between encoder and decoder paths.
    decoder1, decoder2, decoder3, decoder4 : ConvBlock
        Decoder convolutional blocks for feature refinement.
    Conv_1x1 : nn.Conv2d
        Final 1x1 convolutional layer for producing the output segmentation map.

    Methods
    -------
    forward(x)
        Performs the forward pass of the Attention U-Net model.
    """
    
    def __init__(self, in_channels=3, out_channels=1):
        super(AttUNet, self).__init__()

        # Encoding path
        self.encoder1 = ConvBlock(ch_in=in_channels, ch_out=32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = ConvBlock(ch_in=32, ch_out=64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = ConvBlock(ch_in=64, ch_out=128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = ConvBlock(ch_in=128, ch_out=256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock(ch_in=256, ch_out=512)

        # Decoding path with attention
        self.upconv4 = UpConv(ch_in=512, ch_out=256)
        self.attention4 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.decoder4 = ConvBlock(ch_in=512, ch_out=256)

        self.upconv3 = UpConv(ch_in=256, ch_out=128)
        self.attention3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.decoder3 = ConvBlock(ch_in=256, ch_out=128)

        self.upconv2 = UpConv(ch_in=128, ch_out=64)
        self.attention2 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.decoder2 = ConvBlock(ch_in=128, ch_out=64)

        self.upconv1 = UpConv(ch_in=64, ch_out=32)
        self.attention1 = AttentionBlock(F_g=32, F_l=32, F_int=16)
        self.decoder1 = ConvBlock(ch_in=64, ch_out=32)

        # Final output layer
        self.Conv_1x1 = nn.Conv2d(32, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Forward pass for the Attention U-Net model.

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
        x1 = self.encoder1(x)
        x2 = self.pool1(x1)

        x2 = self.encoder2(x2)
        x3 = self.pool2(x2)

        x3 = self.encoder3(x3)
        x4 = self.pool3(x3)

        x4 = self.encoder4(x4)
        x5 = self.pool4(x4)

        # Bottleneck
        x5 = self.bottleneck(x5)

        # Decoding path with attention and skip connections
        d4 = self.upconv4(x5)
        x4 = self.attention4(g=d4, x=x4)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        x3 = self.attention3(g=d3, x=x3)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        x2 = self.attention2(g=d2, x=x2)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        x1 = self.attention1(g=d1, x=x1)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.decoder1(d1)

        # Final 1x1 convolution
        output = self.Conv_1x1(d1)

        return output
