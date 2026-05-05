# models/MSAFI.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class SEBlock3D(nn.Module):
    """3D Squeeze-and-Excitation block for channel-wise attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class Enhanced3DBlock(nn.Module):
    """Enhanced 3D convolution block with residual connection and optional SE attention."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 use_se: bool = True, reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock3D(out_channels, reduction) if use_se else None

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se is not None:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class DilatedConv3DBlock(nn.Module):
    """3D dilated convolution block for enlarged receptive field without extra parameters."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 2):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                              padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extractor using parallel dilated convolutions with different rates."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        quarter = out_channels // 4
        self.dilated1 = DilatedConv3DBlock(in_channels, quarter, dilation=1)
        self.dilated2 = DilatedConv3DBlock(in_channels, quarter, dilation=2)
        self.dilated3 = DilatedConv3DBlock(in_channels, quarter, dilation=4)
        self.dilated4 = DilatedConv3DBlock(in_channels, quarter, dilation=8)
        self.fusion = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.dilated1(x), self.dilated2(x),
                         self.dilated3(x), self.dilated4(x)], dim=1)
        return self.fusion(out)


class SpatialAttention3D(nn.Module):
    """3D spatial attention module highlighting important spatial regions."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return x * attention


class ChannelAttention3D(nn.Module):
    """3D channel attention module using both average and max pooling."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class CBAM3D(nn.Module):
    """3D Convolutional Block Attention Module combining channel and spatial attention."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention3D(channels, reduction)
        self.spatial_attention = SpatialAttention3D(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class LowResolutionEnhancement(nn.Module):
    """Low-resolution feature enhancement via multi-path extraction and CBAM fusion."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.detail_path = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.context_path = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm3d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            CBAM3D(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main_feat = self.main_path(x)
        detail_feat = self.detail_path(x)
        context_feat = self.context_path(x)
        combined = torch.cat([main_feat, detail_feat, context_feat], dim=1)
        return self.fusion(combined)


class EnhancedVideoEncoder(nn.Module):
    """Enhanced video feature encoder with multi-scale extraction, attention, and LR enhancement."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2),
                      padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        self.enhancement = LowResolutionEnhancement(64, 64)
        self.multi_scale = MultiScaleFeatureExtractor(64, 128)
        self.layer1 = Enhanced3DBlock(128, 128, stride=1, use_se=True)
        self.layer2 = Enhanced3DBlock(128, 256, stride=2, use_se=True)
        self.layer3 = Enhanced3DBlock(256, 512, stride=2, use_se=True)
        self.attention1 = CBAM3D(128)
        self.attention2 = CBAM3D(256)
        self.attention3 = CBAM3D(512)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.stem(x)
        x = self.enhancement(x)
        x = self.multi_scale(x)

        features = []
        x = self.attention1(self.layer1(x))
        features.append(x)
        x = self.attention2(self.layer2(x))
        features.append(x)
        x = self.attention3(self.layer3(x))
        features.append(x)

        return x, features


class FeaturePyramidFusion(nn.Module):
    """Feature Pyramid Network for multi-level feature fusion with CBAM."""

    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fusion_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            ))
            self.fusion_convs.append(nn.Sequential(
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            ))

        self.final_fusion = nn.Sequential(
            nn.Conv3d(out_channels * len(in_channels_list), out_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            CBAM3D(out_channels)
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]

        for i in range(len(laterals) - 1, 0, -1):
            target_size = laterals[i - 1].shape[2:]
            upsampled = F.interpolate(laterals[i], size=target_size,
                                      mode='trilinear', align_corners=False)
            laterals[i - 1] = laterals[i - 1] + upsampled

        fused = [conv(lat) for conv, lat in zip(self.fusion_convs, laterals)]

        target_size = fused[0].shape[2:]
        aligned = []
        for feat in fused:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size,
                                     mode='trilinear', align_corners=False)
            aligned.append(feat)

        combined = torch.cat(aligned, dim=1)
        return self.final_fusion(combined)


class AdaptiveClassifier(nn.Module):
    """Adaptive classifier with MLP and dropout."""

    def __init__(self, in_features: int, num_classes: int, dropout_rate: float = 0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class MSAFI(nn.Module):
    """
    MSAFI: Multi-Scale Attention Feature Integration Network.

    Designed for low-resolution fruit quality recognition with:
    - SE channel attention for feature selectivity
    - Multi-scale dilated convolutions for enlarged receptive field
    - CBAM attention for enhanced feature representation
    - Feature Pyramid Network for multi-level fusion
    - Low-resolution enhancement module

    Args:
        num_classes: Number of output classes.
        num_frames: Number of input video frames.
        img_size: Input spatial resolution.
        in_channels: Number of input channels.
    """

    def __init__(self, num_classes: int = 3, num_frames: int = 16,
                 img_size: int = 112, in_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.img_size = img_size

        self.encoder = EnhancedVideoEncoder(in_channels)
        self.fpn = FeaturePyramidFusion([128, 256, 512], 256)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()

        self.projection = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.classifier = AdaptiveClassifier(512, num_classes, dropout_rate=0.4)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, T, H, W).
        Returns:
            Classification logits of shape (B, num_classes).
        """
        _, features = self.encoder(x)
        fused = self.fpn(features)
        pooled = self.global_pool(fused)
        flattened = self.flatten(pooled)
        projected = self.projection(flattened)
        return self.classifier(projected)

    def get_feature_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Return fused feature map and intermediate feature list for visualization."""
        _, features = self.encoder(x)
        fused = self.fpn(features)
        return fused, features