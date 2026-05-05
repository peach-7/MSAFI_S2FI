import torch
import torch.nn as nn
import torch.nn.functional as F

# 轻量级3D卷积模块（适配视频特征提取）
class Basic3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

# 视频特征提取主干
class VideoFeatureEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )

        self.layer1 = Basic3DBlock(32, 64, stride=1)
        self.layer2 = Basic3DBlock(64, 128, stride=2)
        self.layer3 = Basic3DBlock(128, 256, stride=2)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# 多尺度特征融合
class MultiScaleFusion(nn.Module):
    def __init__(self, in_dim=256, embed_dim=512):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        self.proj = nn.Linear(in_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.proj(x)
        x = self.norm(x)
        return x

# 最终可直接用于训练的 DVFLNet
class DVFLNet(nn.Module):
    def __init__(self, num_classes=3, num_frames=16, img_size=112):
        super().__init__()
        self.encoder = VideoFeatureEncoder(in_channels=3)
        self.fusion = MultiScaleFusion(in_dim=256, embed_dim=512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 输入 shape: (B, C, T, H, W)
        x = self.encoder(x)
        x = self.fusion(x)
        x = self.classifier(x)
        return x