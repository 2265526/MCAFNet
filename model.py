import torch
from torch import nn
from config.params import Config
from models.utils import _3DBranch, HSIBranch, LidarEncoder, Cross_atten

config = Config()
dataset_name = config.DATA_PATH.split('/')[-2] if config.DATA_PATH.endswith('/') else config.DATA_PATH.split('/')[-1]
class HSIClassificationMambaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.HSIBranch=HSIBranch(num_bands=self.config.NUM_BANDS)
        self.LidarEncoder = LidarEncoder()
        self.new_3d_branch = _3DBranch(num_bands=self.config.NUM_BANDS)
        self.early_fusion = nn.Sequential(
            nn.Conv2d(self.config.NUM_BANDS + Config.LIDAR_CHANNELS, self.config.NUM_BANDS, 1),
            nn.GELU()
        )
        self.cross_attn = Cross_atten(128, num_heads=4)
        print("Initializing classifier...")
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, Config.NUM_CLASSES)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
    def forward(self, hsi_input, lidar_input):
        # 输入形状:
        # hsi_input:  [B, H, W, num_bands]
        # lidar_input: [B, H, W, 1]
        if dataset_name !='MUUFL':
            lidar_input = lidar_input.unsqueeze(-1)
        combined = torch.cat([hsi_input, lidar_input], dim=-1)  # [B, H, W, num_bands+1]
        combined = combined.permute(0, 3, 1, 2)  # [B, num_bands+1, H, W]
        fused = self.early_fusion(combined) + hsi_input.permute(0, 3, 1, 2)
        # ========== 多模态分支 ==========
        hsi_feat=self.HSIBranch(fused)
        # ==========hsi分支 ==========
        new_3d_feat = self.new_3d_branch(hsi_input)
        # ============ LiDAR分支 =======
        lidar_feat= self.LidarEncoder(lidar_input)
        # ========== 分支融合 ==========
        fused_branch = self.cross_attn(hsi_feat, new_3d_feat)
        fused_branch1 = self.cross_attn(hsi_feat, lidar_feat)
        fused_branch2 = self.cross_attn(new_3d_feat, lidar_feat)
        # ========== 多模态融合 ==========
        final_feat=lidar_feat+fused_branch+fused_branch1+fused_branch2
        final_feat=self.gap(final_feat).view(final_feat.size(0), -1)
        output = self.head(final_feat)
        return output