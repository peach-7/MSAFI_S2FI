# models/MoCLIP_Lite.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCLIP_Lite_Block(nn.Module):
    """Lightweight Transformer block with self-attention and MLP, pure PyTorch."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class MoCLIP_Lite(nn.Module):
    """
    MoCLIP-Lite: lightweight video classification model using 3D-CNN stem + Transformer encoder.

    Input:  (B, 3, num_frames, H, W)
    Output: (B, num_classes)

    Args:
        num_classes: Number of output classes.
        num_frames:  Number of input video frames.
        input_size:  Spatial resolution of input frames.
        embed_dim:   Transformer embedding dimension.
        num_heads:   Number of attention heads.
        num_layers:  Number of Transformer blocks.
        dropout:     Dropout rate.
    """

    def __init__(self, num_classes: int = 3, num_frames: int = 16, input_size: int = 112,
                 embed_dim: int = 256, num_heads: int = 8, num_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.num_frames = num_frames
        self.input_size = input_size
        self.embed_dim = embed_dim

        self.stem = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(128),
            nn.GELU(),
            nn.Conv3d(128, embed_dim, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(embed_dim),
            nn.GELU(),
        )

        self.num_patches = (num_frames // 4) * (input_size // 8) * (input_size // 8)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.transformer_blocks = nn.Sequential(*[
            MoCLIP_Lite_Block(dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.transformer_blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)