# models/C3D.py
import torch.nn as nn

class C3D(nn.Module):
    def __init__(self, num_classes=3, num_frames=16, input_size=112):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), (1, 2, 2)),
            nn.Conv3d(64, 128, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2), (2, 2, 2)),
            nn.Conv3d(128, 256, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2), (2, 2, 2)),
            nn.Conv3d(256, 512, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2), (2, 2, 2)),
            nn.Conv3d(512, 512, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        )
        self.classifier = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
