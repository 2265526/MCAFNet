import torch
from torch import nn
from einops import rearrange
from config.params import Config

class Cross_atten(nn.Module):
    def __init__(self, dim, num_heads):
        super(Cross_atten, self).__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        # Reshape for LayerNorm
        x1 = rearrange(x1, 'b c h w -> b h w c')
        x2 = rearrange(x2, 'b c h w -> b h w c')

        x1 = self.norm1(x1)
        x2 = self.norm2(x2)

        # Reshape back to original shape
        x1 = rearrange(x1, 'b h w c -> b c h w')
        x2 = rearrange(x2, 'b h w c -> b c h w')

        k1 = rearrange(x1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        v1 = rearrange(x1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        k2 = rearrange(x2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        v2 = rearrange(x2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q2 = rearrange(x1, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q1 = rearrange(x2, 'b (head c) h w -> b head h (w c)', head=self.num_heads)

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)

        attn1 = (q1 @ k1.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        out3 = (attn1 @ v1) + q1

        attn2 = (q2 @ k2.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        out4 = (attn2 @ v2) + q2

        out3 = rearrange(out3, 'b head h (w c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out4 = rearrange(out4, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = out3 + out4 + x1 + x2
        return out
class MSFE(nn.Module):
    def __init__(self):
        super().__init__()
        #  多尺度特征融合
        self.multi_scale = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1, dilation=1),
                nn.BatchNorm2d(128),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=2, dilation=2),
                nn.BatchNorm2d(128),
                nn.GELU()),
            # nn.Sequential(
            #     nn.Conv2d(64, 128, 3, padding=3, dilation=3),
            #     nn.BatchNorm2d(128),
            #     nn.GELU()
            # ),
            # nn.Sequential(
            #     nn.Conv2d(64, 128, 3, padding=4, dilation=4),
            #     nn.BatchNorm2d(128),
            #     nn.GELU()
            # )
        ])
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=128*2, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # 轻量级通道注意力
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, 128, 1),
            nn.Sigmoid()
        )
    def forward(self,x):
            # 多尺度融合
            scale_features = []
            for block in self.multi_scale:
                scale_features.append(block(x))
            fused = torch.cat(scale_features, dim=1)
            fused = self.conv2d(fused)
            #通道注意力增强
            attn_weights = self.attn(fused)
            out = fused * attn_weights
            return out
class HSIBranch(nn.Module):
    def __init__(self, num_bands):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_bands, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        #深度可分离残差块
        self.dw_conv_block = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, groups=64),  # 深度卷积
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, 1),  # 逐点扩展
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.MSFE=MSFE()#多尺度光谱增强模块
    def forward(self, x):
        x=self.conv1(x)
        x1 = self.dw_conv_block(x)
        #print('x1', x1.shape)
        out= self.MSFE(x1)
        return out  # [B,128，11,11]
class LidarEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入处理层
        self.conv1 = nn.Sequential(
            nn.Conv2d(Config.LIDAR_CHANNELS, 32, 3, padding=1),#MUUFL 2
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        # 特征提取
        x = self.conv1(x)
        x = self.conv2(x)
        return x
class EfficientScanPath(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.patches=Config.PATCH_SIZE
        self.norm = nn.LayerNorm([in_dim, 2, self.patches, self.patches])
        # 双向共享参数
        self.A = nn.Parameter(torch.randn(in_dim, in_dim))
        self.B = nn.Parameter(torch.randn(in_dim, in_dim))
    def _bidirectional_scan(self, x):
        B, C, D, H, W = x.shape
        x_seq = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, D, C)
        # 正向扫描
        shifted = torch.roll(x_seq, 1, dims=1)
        shifted[:, 0] = 0
        forward = torch.sigmoid(torch.einsum('bsc,cd->bsd', x_seq, self.A) +
                                torch.einsum('bsc,cd->bsd', shifted, self.B))
        # 反向扫描
        reversed_seq = torch.flip(x_seq, [1])
        shifted_rev = torch.roll(reversed_seq, 1, dims=1)
        shifted_rev[:, 0] = 0
        backward = torch.flip(
            torch.sigmoid(torch.einsum('bsc,cd->bsd', reversed_seq, self.A) +
                          torch.einsum('bsc,cd->bsd', shifted_rev, self.B)),
            [1]
        )
        # 合并双向结果
        combined = (forward + backward) / 2
        # 恢复形状
        return combined.view(B, H, W, D, C).permute(0, 4, 3, 1, 2)
    def forward(self, x):
        residual=x
        x=self._bidirectional_scan(x)
        x = self.norm(x)
        x=x+residual
        return x
class _3DBranch(nn.Module):
    def __init__(self, num_bands):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(num_bands, num_bands, 3, padding=1, groups=num_bands),
            nn.Conv3d(num_bands, 128, 1),
            nn.BatchNorm3d(128),
            nn.GELU(),
            nn.MaxPool3d(1)
        )
        self.ss3d = EfficientScanPath(in_dim=128)
    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2).unsqueeze(2).expand(-1, -1, 2, -1, -1)  # [B,128,2,11,11]    # 转换为3D输入
        x_3d=self.conv3d(x)
        ss_out = self.ss3d(x_3d).mean(2) # [B,128,11,11]
        return ss_out
