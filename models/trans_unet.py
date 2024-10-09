# Implement the TransUNet model
import torch
import torch.nn as nn
from models.transformer_encoder import TransformerEncoder


class TransUNet(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        num_classes=2,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1,
    ):
        super(TransUNet, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size**2

        self.patch_embed = nn.Conv2d(
            3, dim, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.pos_drop = nn.Dropout(p=emb_dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoder(
                    dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout
                )
                for _ in range(3)
            ]
        )

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
