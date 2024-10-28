import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseConvBlock(nn.Module):
    """
    A depthwise convolutional block that consists of depthwise and pointwise convolutions
    followed by batch normalization and ReLU activation.

    Parameters
    ----------
    ch_in : int
        Number of input channels.
    ch_out : int
        Number of output channels.

    Attributes
    ----------
    depthwise : nn.Conv2d
        Depthwise convolution with input channels equal to the number of groups.
    pointwise : nn.Conv2d
        Pointwise convolution to combine depthwise convolutions.
    bn : nn.BatchNorm2d
        Batch normalization applied after the convolutions.
    relu : nn.ReLU
        ReLU activation function.

    Methods
    -------
    forward(x)
        Forward pass of the depthwise convolutional block.
    """

    def __init__(self, ch_in, ch_out):
        super(DepthwiseConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in, bias=False)
        self.pointwise = nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass for the DepthwiseConvBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, H, W) where:
            - N is the batch size.
            - C is the number of channels.
            - H is the height of the feature map.
            - W is the width of the feature map.

        Returns
        -------
        torch.Tensor
            Output tensor after depthwise and pointwise convolutions.
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TransformerBlock(nn.Module):
    """
    A transformer block that consists of multi-head self-attention followed by a feed-forward MLP.

    Parameters
    ----------
    dim : int
        Dimension of the input and output features.
    heads : int
        Number of attention heads.
    mlp_dim : int
        Dimension of the feed-forward MLP.
    dropout : float, optional
        Dropout rate applied to the attention and MLP layers (default is 0.1).

    Attributes
    ----------
    attn : nn.MultiheadAttention
        Multi-head self-attention layer.
    norm1 : nn.LayerNorm
        Layer normalization applied after the attention layer.
    mlp : nn.Sequential
        Feed-forward MLP with GELU activation and dropout.
    norm2 : nn.LayerNorm
        Layer normalization applied after the feed-forward MLP.

    Methods
    -------
    forward(x)
        Forward pass of the transformer block.
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
            Input tensor of shape (S, B, C) where:
            - S is the sequence length (flattened spatial dimensions).
            - B is the batch size.
            - C is the number of feature channels.

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
    TransUNet architecture for image segmentation tasks. Combines convolutional encoders with 
    transformer blocks to capture both local and global dependencies.

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels (default is 3).
    out_channels : int, optional
        Number of output channels (default is 1).
    transformer_dim : int, optional
        Dimension of the transformer features (default is 256).
    num_heads : int, optional
        Number of attention heads in each transformer block (default is 4).
    mlp_dim : int, optional
        Dimension of the feed-forward MLP in the transformer block (default is 512).
    transformer_depth : int, optional
        Number of transformer blocks in the bottleneck (default is 6).

    Attributes
    ----------
    encoder1, encoder2, encoder3, encoder4 : nn.Module
        Encoder blocks using depthwise convolutional layers.
    pool1, pool2, pool3, pool4 : nn.MaxPool2d
        Pooling layers for downsampling the feature maps.
    encoder_projection : nn.Conv2d
        Convolutional layer for projecting the number of channels from the encoder output 
        to the transformer dimension.
    transformer_blocks : nn.Sequential
        Sequence of transformer blocks in the bottleneck.
    upconv4, upconv3, upconv2, upconv1 : nn.ConvTranspose2d
        Upsampling layers for increasing the feature map size in the decoder path.
    decoder4, decoder3, decoder2, decoder1 : nn.Module
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

        # Encoding path
        self.encoder1 = DepthwiseConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = DepthwiseConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = DepthwiseConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = DepthwiseConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Transformer bottleneck
        self.flatten = nn.Flatten(2)
        self.position_embeddings = nn.Parameter(torch.randn(1, transformer_dim, 512 // 16 * 512 // 16))

        # Add encoder_projection to project 512 channels to transformer_dim (256 in this case)
        self.encoder_projection = nn.Conv2d(512, transformer_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(transformer_dim, num_heads, mlp_dim) for _ in range(transformer_depth)]
        )

        # Decoding path
        self.upconv4 = nn.ConvTranspose2d(transformer_dim, 256, kernel_size=2, stride=2)
        self.decoder4 = DepthwiseConvBlock(256 + 256, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = DepthwiseConvBlock(128 + 128, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = DepthwiseConvBlock(64 + 64, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = DepthwiseConvBlock(32, 32)

        # Final 1x1 conv to get the output
        self.Conv_1x1 = nn.Conv2d(32, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Forward pass for the TransUNet model.

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
            Output tensor of shape (N, out_channels, 128, 128).
        """
        # Encoding path
        x1 = self.encoder1(x)
        x2 = self.pool1(x1)
        x2 = self.encoder2(x2)

        x3 = self.pool2(x2)
        x3 = self.encoder3(x3)

        x4 = self.pool3(x3)
        x4 = self.encoder4(x4)

        # Project the channels from 512 to transformer_dim (256)
        x_projected = self.encoder_projection(x4)  # (B, transformer_dim, H//8, W//8)

        # Get the spatial dimensions (H and W) of x_projected
        B, C, H, W = x_projected.shape

        # Flatten and pass through transformer blocks
        x_flattened = self.flatten(x_projected)  # (B, transformer_dim, H*W)

        # Add positional embeddings
        pos_embed = self.position_embeddings[:, :, :H * W]  # Ensure size matches with x_flattened
        x_flattened += pos_embed

        x_transformed = x_flattened.permute(2, 0, 1)  # (H*W, B, transformer_dim)

        for block in self.transformer_blocks:
            x_transformed = block(x_transformed)

        # Unflatten back to original dimensions (adjusted for transformer_dim)
        x_transformed = x_transformed.permute(1, 2, 0).view(B, C, H, W)

        # Decoding path with skip connections
        d4 = self.upconv4(x_transformed)
        d4 = torch.cat((d4, x3), dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, x2), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, x1), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = self.decoder1(d1)

        output = self.Conv_1x1(d1)

        output = F.interpolate(output, size=(128, 128), mode='bilinear', align_corners=False)

        return output

