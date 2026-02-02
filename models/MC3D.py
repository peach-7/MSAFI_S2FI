# models/MC3D.py
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out

class MC3D(nn.Module):
    def __init__(self, num_classes=3, num_frames=16, input_size=112, num_clips=2):
        super().__init__()
        self.num_frames = num_frames
        self.input_size = input_size
        self.num_clips = num_clips
        self.clip_len = num_frames // num_clips
        self.features = nn.Sequential(
            nn.Conv3d(3, 32, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), (1, 2, 2)),
            ResidualBlock(32, 64, stride=1),
            ResidualBlock(64, 64, stride=1),
            nn.MaxPool3d((1, 2, 2), (1, 2, 2)),
            ResidualBlock(64, 128, stride=1),
            ResidualBlock(128, 128, stride=1),
            nn.MaxPool3d((1, 2, 2), (1, 2, 2)),
            ResidualBlock(128, 256, stride=1),
            ResidualBlock(256, 256, stride=1),
            nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        )
        self.fusion = nn.Sequential(
            nn.Linear(200704, 1024),
            nn.BatchNorm1d(1024), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _split_into_clips(self, x):
        B, C, T, H, W = x.shape
        clip_len = T // self.num_clips
        clips = []
        for i in range(self.num_clips):
            start = i * clip_len
            end = start + clip_len if i < self.num_clips - 1 else T
            clips.append(x[:, :, start:end, :, :])
        return clips

    def forward(self, x):
        clips = self._split_into_clips(x)
        clip_features = []
        for clip in clips:
            feat = self.features(clip)
            feat = feat.view(feat.size(0), -1)
            clip_features.append(feat)
        fused_feat = torch.cat(clip_features, dim=1)
        return self.fusion(fused_feat)
