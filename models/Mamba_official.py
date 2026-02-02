# models/Mamba/Mamba_official.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
            bias=False
        )
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        self.A_log = nn.Parameter(torch.log(
            torch.arange(1, d_state + 1, dtype=torch.float32)
        ).repeat(d_inner, 1))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x):
        B, T, C = x.shape
        x_and_res = self.in_proj(x)  # [B, T, 2*d_inner]
        x, res = x_and_res.chunk(2, dim=-1)  # [B, T, d_inner] each
        x = x.permute(0, 2, 1)  # [B, d_inner, T]
        x = self.conv1d(x)[:, :, :T]  # [B, d_inner, T]
        x = x.permute(0, 2, 1)  # [B, T, d_inner]
        x = self.act(x)
        x = x + res
        x = self.out_proj(x)  # [B, T, C]
        return x

class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, 4, 4))
        )

    def forward(self, x):
        return self.conv3d(x)

class Mamba_official(nn.Module):
    def __init__(self, num_classes=3, num_frames=16, input_size=112):
        super().__init__()
        self.num_classes = num_classes
        self.video_encoder = VideoEncoder()
        with torch.no_grad():
            dummy = torch.zeros(2, 3, num_frames, input_size, input_size)
            feat = self.video_encoder(dummy)
            _, c, t, h, w = feat.shape
            flat_dim = c * h * w
            self.actual_time_steps = t

        self.feature_proj = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.LayerNorm(256),
            nn.Dropout(0.2)
        )

        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model=256),
            MambaBlock(d_model=256),
            MambaBlock(d_model=256),
            MambaBlock(d_model=256),
        ])

        self.fusion = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, return_features=False):
        B, C, T, H, W = x.shape
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        video_feat = self.video_encoder(x)
        _, c, t, h, w = video_feat.shape
        video_feat = video_feat.permute(0, 2, 1, 3, 4).reshape(B, t, c * h * w)
        seq_feat = self.feature_proj(video_feat)  # [B, T, 256]
        for layer in self.mamba_layers:
            seq_feat = layer(seq_feat)

        seq_feat = seq_feat.permute(0, 2, 1)  # [B, 256, T]
        avg_feat = torch.mean(seq_feat, dim=2, keepdim=True)  # [B, 256, 1]
        max_feat, _ = torch.max(seq_feat, dim=2, keepdim=True)  # [B, 256, 1]
        std_feat = torch.std(seq_feat, dim=2, keepdim=True)  # [B, 256, 1]
        fused = torch.cat([avg_feat, max_feat, std_feat], dim=2)  # [B, 256, 3]
        fused = fused.view(B, -1)  # [B, 256*3]
        fused = self.fusion(fused)

        if return_features:
            output = self.classifier(fused)
            return output, fused

        return self.classifier(fused)
