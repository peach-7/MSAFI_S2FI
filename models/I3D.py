# models/I3D.py
import torch
import torch.nn as nn

class Conv3DSimple(nn.Conv3d):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, bias=False):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        super().__init__(in_channels=in_planes, out_channels=out_planes,
                         kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self.conv1 = nn.Conv3d(inplanes, planes, 1, 1, 0, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv3d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, 1, 1, 0, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class I3D(nn.Module):
    def __init__(self, num_classes=3, block=Bottleneck, layers=[3, 4, 6, 3], norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self.inplanes = 64
        self.conv1 = Conv3DSimple(3, 64, 7, (1, 2, 2), (3, 3, 3), False)
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
        for m in self.modules():
            if isinstance(m, Conv3DSimple):
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
                Conv3DSimple(self.inplanes, planes * block.expansion, 1, stride, 0, False),
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
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)
