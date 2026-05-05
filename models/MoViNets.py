# models/MoViNets.py
import torch
import torch.nn as nn


class MoViNetBlock(nn.Module):
    """MoViNet residual block with depthwise separable 3D convolution."""

    def __init__(self, in_channels: int, out_channels: int,
                 stride: tuple = (1, 1, 1), expand_ratio: int = 4):
        super().__init__()
        hidden_dim = in_channels * expand_ratio

        self.conv1 = nn.Conv3d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(hidden_dim)
        self.conv2 = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3,
                               stride=stride, padding=1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm3d(hidden_dim)
        self.conv3 = nn.Conv3d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.act = nn.SiLU()

        self.shortcut = nn.Identity()
        if stride != (1, 1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.act(x + residual)


class MoViNets(nn.Module):
    """
    MoViNets: Mobile Video Network for lightweight 3D video classification.

    Input:  (B, 3, num_frames, H, W)
    Output: (B, num_classes)

    Args:
        num_classes: Number of output classes.
        num_frames:  Number of input video frames.
        input_size:  Spatial resolution.
        width_mult:  Width multiplier for channel scaling.
    """

    def __init__(self, num_classes: int = 3, num_frames: int = 16,
                 input_size: int = 112, width_mult: float = 1.0):
        super().__init__()

        self.cfgs = [
            [16, 16, (1, 2, 2), 1],
            [16, 32, (1, 2, 2), 2],
            [32, 48, (1, 2, 2), 2],
            [48, 64, (1, 1, 1), 3],
            [64, 96, (1, 1, 1), 2],
        ]

        self.stem = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.SiLU(),
        )

        layers = []
        in_ch = 16
        for in_c, out_c, stride, repeat in self.cfgs:
            out_c = int(out_c * width_mult)
            for i in range(repeat):
                s = stride if i == 0 else (1, 1, 1)
                layers.append(MoViNetBlock(in_ch, out_c, stride=s))
                in_ch = out_c
        self.blocks = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Conv3d(in_ch, 320, kernel_size=1, bias=False),
            nn.BatchNorm3d(320),
            nn.SiLU(),
        )
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(320, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        return self.classifier(x)