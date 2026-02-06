# transformer_based_on_i3d.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --- 来自原始 Transformer.py 的增强型组件 ---

class MultiHeadAttentionEnhanced(nn.Module):
    """增强型多头自注意力机制"""

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

        # 增强门控
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, D = x.shape  # Batch, Time, Dim

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
    """增强型前馈网络"""

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
    """增强型 Transformer 编码器层"""

    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttentionEnhanced(d_model, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardEnhanced(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)
        # 残差缩放
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, mask=None):
        residual = self.residual_scale * self.attn(self.norm1(x), mask)
        x = x + self.dropout1(residual)

        residual = self.ffn(self.norm2(x))
        x = x + self.dropout2(residual)
        return x


class LearnedPositionalEncoding(nn.Module):
    """学习的位置编码"""

    def __init__(self, d_model, max_len=2048):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.1)

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            # 如果序列长度超过预设，则动态扩展
            expanded_pe = torch.randn(1, x.size(1), self.pe.size(2)).to(x.device) * 0.1
            return x + expanded_pe
        return x + self.pe[:, :x.size(1)]


# --- 来自原始 I3D.py 的核心组件 (Conv3DSimple, Bottleneck) ---

class Conv3DSimple(nn.Conv3d):
    """简单的 3D 卷积块，用于 I3D 骨干"""

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, bias=False):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        super().__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )


class Bottleneck(nn.Module):
    """I3D 使用的瓶颈块"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        # 1x1 conv 降维
        self.conv1 = nn.Conv3d(inplanes, planes, 1, 1, 0, bias=False)
        self.bn1 = norm_layer(planes)
        # 3x3x3 conv 提取时空特征
        self.conv2 = nn.Conv3d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = norm_layer(planes)
        # 1x1 conv 升维
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


# --- 新的混合模型定义 ---
class TransformerBasedOnI3D(nn.Module):
    """
    结合 I3D 骨干和增强型 Transformer 的混合模型。
    利用 I3D 进行强大的时空特征提取，然后用 Transformer 对特征序列进行高级建模。
    """

    def __init__(self, num_classes=3, num_frames=16, input_size=112):
        super().__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.input_size = input_size

        # --- I3D 骨干构建 ---
        norm_layer = nn.BatchNorm3d
        self.conv1 = Conv3DSimple(3, 64, 7, (1, 2, 2), (3, 3, 3), False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1))

        # 构建各个层，注意通道数和步长
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1, norm_layer=norm_layer)  # 输出 256
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2, norm_layer=norm_layer)  # 输出 512
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2, norm_layer=norm_layer)  # 输出 1024
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2, norm_layer=norm_layer)  # 输出 2048

        # 最终的池化层
        self.avgpool = nn.AdaptiveAvgPool3d((None, 4, 4))  # 固定空间尺寸为 (4, 4)，时间维度保持

        # 将 I3D 骨干的各部分组合成一个顺序容器
        self.i3d_backbone = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.maxpool,
            self.layer1, self.layer2, self.layer3, self.layer4,
            self.avgpool
        )

        # --- Transformer 部分 ---
        # 通过一次推理确定中间维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, num_frames, input_size, input_size)
            i3d_out = self.i3d_backbone(dummy_input)
            _, c, t, h, w = i3d_out.shape
            self.flat_dim = c * h * w  # 例如 2048 * 4 * 4 = 32768
            self.time_steps = t  # T dimension after I3D

        # 特征投影层
        self.feature_proj = nn.Sequential(
            nn.Linear(self.flat_dim, 512),  # 例如，投影到 512 维
            nn.LayerNorm(512),
            nn.Dropout(0.2)
        )

        # 增强型 Transformer 编码器
        d_model = 512  # 与投影层输出维度一致
        self.transformer = nn.Sequential(
            LearnedPositionalEncoding(d_model, max_len=2048),  # 加入位置编码
            *[TransformerEncoderLayerEnhanced(
                d_model=d_model,
                num_heads=16,  # 可调参数
                d_ff=2048,  # 可调参数
                dropout=0.2  # 可调参数
            ) for _ in range(6)]  # 6 层编码器
        )

        # 时序池化和多尺度池化
        self.multi_scale_pool = nn.ModuleList([
            nn.AdaptiveAvgPool1d(1),
            nn.AdaptiveMaxPool1d(1)
        ])

        # 注意力门控 (修正：为 d_model 维度的每个特征生成一个权重)
        self.attention_gate = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.SiLU(),
            nn.Linear(256, d_model),  # 输出维度改为 d_model
            nn.Sigmoid()
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.3),
            nn.Linear(d_model, 1024),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes)
        )

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        """辅助函数，构建 I3D 的残差层"""
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d

        # 确定输入通道数
        if planes == 64:  # layer1, 接收 conv1 (64 channels) 的输出
            inplanes = 64
        elif planes == 128:  # layer2, 接收 layer1 (64*4=256 channels) 的输出
            inplanes = 64 * Bottleneck.expansion
        elif planes == 256:  # layer3, 接收 layer2 (128*4=512 channels) 的输出
            inplanes = 128 * Bottleneck.expansion
        elif planes == 512:  # layer4, 接收 layer3 (256*4=1024 channels) 的输出
            inplanes = 256 * Bottleneck.expansion
        else:
            raise ValueError(f"Unexpected planes value: {planes}")

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv3DSimple(inplanes, planes * block.expansion, 1, stride, 0, False),
                norm_layer(planes * block.expansion)
            )

        layers = [block(inplanes, planes, stride, downsample, norm_layer)]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        B, C, T, H, W = x.shape

        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        # 1. I3D 骨干特征提取
        # Shape: (B, 3, T, H, W) -> (B, 2048, T', 4, 4) (取决于具体的 strides)
        conv_features = self.i3d_backbone(x)
        B, C_i3d, T_i3d, H_i3d, W_i3d = conv_features.shape

        # 2. 重塑为序列
        # Shape: (B, T', C_i3d * H_i3d * W_i3d)
        x_seq = conv_features.permute(0, 2, 1, 3, 4).reshape(B, T_i3d, self.flat_dim)

        # 3. 投影到 Transformer 维度
        # Shape: (B, T', 512)
        x_proj = self.feature_proj(x_seq)

        # 4. Transformer 编码
        # Shape: (B, T', 512)
        x_trans = self.transformer(x_proj)

        # 5. 调整维度为 (B, 512, T') 以便于池化
        x_trans_t = x_trans.permute(0, 2, 1)  # (B, 512, T')

        # 6. 多尺度时序池化
        avg_pooled = self.multi_scale_pool[0](x_trans_t).squeeze(-1)  # (B, 512)
        max_pooled = self.multi_scale_pool[1](x_trans_t).squeeze(-1)  # (B, 512)
        x_pooled = avg_pooled + max_pooled  # (B, 512)

        # 7. 应用注意力门控 (修正后，形状匹配)
        # attention_weights: (B, 512)
        attention_weights = self.attention_gate(x_pooled)
        # x_pooled: (B, 512), attention_weights: (B, 512) -> (B, 512)
        x_enhanced = x_pooled * attention_weights

        # 8. 最终分类
        output = self.classifier(x_enhanced)  # (B, num_classes)

        if return_features:
            return output, x_enhanced  # 返回 logits 和增强后的特征向量

        return output

# Example usage:
# model = TransformerBasedOnI3D(num_classes=3, num_frames=16, input_size=112)
# print(model)
# x = torch.randn(2, 3, 16, 112, 112)
# y = model(x)
# print(y.shape) # Should be [2, 3]