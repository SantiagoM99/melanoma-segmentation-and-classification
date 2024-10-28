import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseConvBlock(nn.Module):
    """
    Depthwise convolutional block used in the TransUNet model for efficient feature extraction.

    This block employs depthwise and pointwise convolutions followed by batch normalization and ReLU activation.

    Parameters
    ----------
    ch_in : int
        Number of input channels.
    ch_out : int
        Number of output channels.

    Attributes
    ----------
    conv1 : nn.Conv2d
        Depthwise convolution with `ch_in` groups.
    pointwise1 : nn.Conv2d
        Pointwise convolution for combining depthwise convolutions.
    bn1 : nn.BatchNorm2d
        Batch normalization applied after the first convolution.
    conv2 : nn.Conv2d
        Depthwise convolution with `ch_out` groups.
    pointwise2 : nn.Conv2d
        Pointwise convolution for combining depthwise convolutions.
    bn2 : nn.BatchNorm2d
        Batch normalization applied after the second convolution.
    relu : nn.ReLU
        ReLU activation function.

    Methods
    -------
    forward(x)
        Performs the forward pass of the depthwise convolutional block.
    """

    def __init__(self, ch_in, ch_out):
        super(DepthwiseConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in, bias=False)
        self.pointwise1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch_out)
        
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, groups=ch_out, bias=False)
        self.pointwise2 = nn.Conv2d(ch_out, ch_out, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch_out)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass for the DepthwiseConvBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor after depthwise and pointwise convolutions.
        """
        x = self.conv1(x)
        x = self.pointwise1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.pointwise2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block used in the TransUNet model.

    This block employs multi-head self-attention followed by a feed-forward MLP with GELU activation.

    Parameters
    ----------
    dim : int
        Dimension of the input and output features.
    heads : int
        Number of attention heads.
    mlp_dim : int
        Dimension of the feed-forward MLP.
    dropout : float, optional
        Dropout rate applied to the attention and MLP layers. Default is 0.1.

    Attributes
    ----------
    attn : nn.MultiheadAttention
        Multi-head self-attention layer.
    norm1 : nn.LayerNorm
        Layer normalization applied after attention.
    mlp : nn.Sequential
        Feed-forward MLP with GELU activation and dropout.
    norm2 : nn.LayerNorm
        Layer normalization applied after the feed-forward MLP.

    Methods
    -------
    forward(x)
        Performs the forward pass of the transformer block.
    """

    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Forward pass for the TransformerBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, N, C) where:
            - B is the batch size
            - N is the number of patches
            - C is the feature dimension

        Returns
        -------
        torch.Tensor
            Output tensor after self-attention and feed-forward layers.
        """
        attn_out, _ = self.attn(x, x, x)  # Self-attention
        x = self.norm1(attn_out + x)  # Add & Norm

        mlp_out = self.mlp(x)
        x = self.norm2(mlp_out + x)  # Add & Norm

        return x


class TransUNet(nn.Module):
    """
    TransUNet architecture for image segmentation tasks.

    This model combines convolutional encoders with transformer blocks in the bottleneck to capture
    both local and global dependencies for image segmentation.

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels. Default is 3.
    out_channels : int, optional
        Number of output channels. Default is 1.
    transformer_dim : int, optional
        Dimension of the transformer features. Default is 256.
    num_heads : int, optional
        Number of attention heads in each transformer block. Default is 4.
    mlp_dim : int, optional
        Dimension of the feed-forward MLP in the transformer block. Default is 512.
    transformer_depth : int, optional
        Number of transformer blocks in the bottleneck. Default is 6.

    Attributes
    ----------
    encoder1, encoder2, encoder3, encoder4 : nn.Sequential
        Encoder blocks using depthwise convolutional layers.
    pool1, pool2, pool3, pool4 : nn.MaxPool2d
        Pooling layers for downsampling the feature maps.
    transformer_blocks : nn.Sequential
        Sequence of transformer blocks in the bottleneck.
    upconv1, upconv2, upconv3, upconv4 : nn.ConvTranspose2d
        Upsampling layers for increasing the feature map size in the decoder path.
    decoder1, decoder2, decoder3, decoder4 : nn.Sequential
        Decoder blocks using depthwise convolutional layers.
    Conv_1x1 : nn.Conv2d
        Final 1x1 convolutional layer for producing the output segmentation map.

    Methods
    -------
    forward(x)
        Performs the forward pass of the TransUNet model.
    """

    def __init__(self, in_channels=3, out_channels=1, transformer_dim=256, num_heads=4, mlp_dim=512, transformer_depth=6):
        super(TransUNet, self).__init__()

        # Encoding path using DepthwiseConvBlock
        self.encoder1 = nn.Sequential(DepthwiseConvBlock(ch_in=in_channels, ch_out=64))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = nn.Sequential(DepthwiseConvBlock(ch_in=64, ch_out=128))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = nn.Sequential(DepthwiseConvBlock(ch_in=128, ch_out=256))
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = nn.Sequential(DepthwiseConvBlock(ch_in=256, ch_out=512))
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Transformer bottleneck
        self.flatten = nn.Flatten(2)  # Convert to [B, C, H*W]
        self.transpose = lambda x: x.permute(2, 0, 1)  # Permute to [H*W, B, C]

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(transformer_dim, num_heads, mlp_dim) for _ in range(transformer_depth)]
        )
        self.unflatten = nn.Unflatten(0, (32, 32))  # Adjust this based on input dimensions

        # Decoding path (use transposed convolutions and concatenation like in U-Net)
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(DepthwiseConvBlock(ch_in=512, ch_out=256))

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(DepthwiseConvBlock(ch_in=256, ch_out=128))

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(DepthwiseConvBlock(ch_in=128, ch_out=64))

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(DepthwiseConvBlock(ch_in=64, ch_out=32))

        # Final 1x1 conv to get the output
        self.Conv_1x1 = nn.Conv2d(32, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Forward pass for the TransUNet model.

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
        x1 = self.encoder1(x)  # (B, 64, H, W)
        x2 = self.pool1(x1)  # (B, 64, H//2, W//2)
        x2 = self.encoder2(x2)  # (B, 128, H//2, W//2)

        x3 = self.pool2(x2)  # (B, 128, H//4, W//4)
        x3 = self.encoder3(x3)  # (B, 256, H//4, W//4)

        x4 = self.pool3(x3)  # (B, 256, H//8, W//8)
        x4 = self.encoder4(x4)  # (B, 512, H//8, W//8)

        # Flatten and pass through transformer blocks
        x_flattened = self.flatten(x4)
        x_transformed = self.transpose(x_flattened)
        for block in self.transformer_blocks:
            x_transformed = block(x_transformed)
        x_transformed = self.unflatten(x_transformed)

        # Decoding path with skip connections
        d4 = self.upconv4(x_transformed)  # (B, 256, H//8, W//8)
        d4 = torch.cat((x3, d4), dim=1)  # Concatenate with encoder3 output
        d4 = self.decoder4(d4)  # (B, 256, H//8, W//8)

        d3 = self.upconv3(d4)  # (B, 128, H//4, W//4)
        d3 = torch.cat((x2, d3), dim=1)  # Concatenate with encoder2 output
        d3 = self.decoder3(d3)  # (B, 128, H//4, W//4)

        d2 = self.upconv2(d3)  # (B, 64, H//2, W//2)
        d2 = torch.cat((x1, d2), dim=1)  # Concatenate with encoder1 output
        d2 = self.decoder2(d2)  # (B, 64, H//2, W//2)

        d1 = self.upconv1(d2)  # (B, 32, H, W)
        d1 = self.decoder1(d1)  # (B, 32, H, W)

        # Final 1x1 conv to get the output
        output = self.Conv_1x1(d1)  # (B, out_channels, H, W)

        return output
