import torch
import torch.nn as nn
import torch.nn.functional as F

class R3DBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self.conv1 = nn.Conv3d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        # x shape: [B, T, C] -> [T, B, C] for MultiheadAttention
        x = x.permute(1, 0, 2)
        attn_output, _ = self.attention(x, x, x)
        # Convert back to [B, T, C]
        return attn_output.permute(1, 0, 2)

class R3D(nn.Module):
    def __init__(self, block=R3DBasicBlock, layers=[3, 4, 6, 4], num_classes=3, norm_layer=None, embed_dim=512, num_heads=8):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self.inplanes = 64
        self.conv1 = nn.Conv3d(3, 64, (3, 7, 7), (1, 2, 2), (1, 3, 3), bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1))
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                norm_layer(planes * block.expansion)
            )
        layers = [block(self.inplanes, planes, stride, downsample, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten the output: [B, 512]
        x = self.dropout(x)
        x = x.unsqueeze(1)  # Add time dimension: [B, 1, 512]
        x = self.attention(x)  # Apply attention: [B, 1, 512]
        x = x.squeeze(1)  # Remove time dimension: [B, 512]
        return self.fc(x)
