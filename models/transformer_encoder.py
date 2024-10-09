# Implement the TransformerEncoder class in models/transformer_encoder.py:
import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(dim),
                        nn.MultiheadAttention(dim, heads=heads, dropout=dropout),
                        nn.Dropout(dropout),
                        nn.LayerNorm(dim),
                        nn.Linear(dim, mlp_dim),
                        nn.ReLU(),
                        nn.Linear(mlp_dim, dim),
                        nn.Dropout(dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn_norm, attn, attn_drop, mlp_norm, mlp, mlp_act, mlp_drop in self.layers:
            x = x + attn_drop(attn(attn_norm(x, x, x)))
            x = x + mlp_drop(mlp(mlp_act(mlp_norm(x))))
        return x
