# models/Transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttentionEnhanced(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)

        scores = torch.einsum('bqid,bqjd->bqij', q, k) * self.scaling

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.einsum('bqij,bqjd->bqid', attn_weights, v)
        attn_output = attn_output.contiguous().view(B, T, D)

        gate = self.gate(attn_output)
        output = self.out_proj(attn_output * gate)

        return output


class FeedForwardEnhanced(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderLayerEnhanced(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttentionEnhanced(d_model, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardEnhanced(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, mask=None):
        residual = self.residual_scale * self.attn(self.norm1(x), mask)
        x = x + self.dropout1(residual)

        residual = self.ffn(self.norm2(x))
        x = x + self.dropout2(residual)

        return x


class TransformerEncoderEnhanced(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff=None, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayerEnhanced(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        self.pos_embed = LearnedPositionalEncoding(d_model, max_len=2048)

    def forward(self, x, mask=None):
        x = self.pos_embed(x)

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.1)

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            expanded_pe = torch.randn(1, x.size(1), self.pe.size(2)).to(x.device) * 0.1
            return x + expanded_pe
        return x + self.pe[:, :x.size(1)]


class Transformer(nn.Module):
    def __init__(self, num_classes=3, num_frames=16, input_size=112):
        super().__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.input_size = input_size

        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 48, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(48, 96, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(96, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1),
                      groups=32, stride=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((None, 4, 4))
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, num_frames, input_size, input_size)
            conv_out = self.conv3d(dummy)
            _, c, t, h, w = conv_out.shape
            flat_dim = c * h * w
            self.time_steps = t

        self.feature_proj = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.2)
        )

        self.transformer = TransformerEncoderEnhanced(
            num_layers=6,
            d_model=512,
            num_heads=16,  # 更多注意力头
            d_ff=2048,
            dropout=0.2
        )

        self.time_pool = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.multi_scale_pool = nn.ModuleList([
            nn.AdaptiveAvgPool1d(1),
            nn.AdaptiveMaxPool1d(1)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes)
        )

        self.attention_gate = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, return_features=False):
        B, C, T, H, W = x.shape

        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        conv_features = self.conv3d(x)
        _, C2, T2, H2, W2 = conv_features.shape

        x_seq = conv_features.permute(0, 2, 1, 3, 4).reshape(B, T2, C2 * H2 * W2)

        x_proj = self.feature_proj(x_seq)

        x_trans = self.transformer(x_proj)

        x_trans = x_trans.permute(0, 2, 1)

        avg_pool = self.multi_scale_pool[0](x_trans).squeeze(-1)
        max_pool = self.multi_scale_pool[1](x_trans).squeeze(-1)
        x_pooled = avg_pool + max_pool

        attention_weights = self.attention_gate(x_pooled)
        x_enhanced = x_pooled * attention_weights.squeeze(-1).unsqueeze(1)

        output = self.classifier(x_enhanced)

        if return_features:
            return output, x_enhanced.mean(dim=1)

        return output
