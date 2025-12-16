"""
轻量化人脸投影层：基于3D关键点实现2D→3D→2D投影
参考ViewDiff的投影层思想，但简化为人脸特定
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LightweightFaceProjectionLayer(nn.Module):
    """
    轻量化人脸投影层：基于3D关键点实现视角变换
    
    核心思想：
    - 使用3D关键点作为几何先验
    - 通过轻量级体素网格实现3D特征表示
    - 基于目标姿态进行体积渲染
    """
    def __init__(self, feature_dim, num_keypoints=5, voxel_resolution=16):
        """
        Args:
            feature_dim: 特征维度
            num_keypoints: 3D关键点数量（默认5，InsightFace）
            voxel_resolution: 体素网格分辨率（默认16，平衡精度和计算成本）
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_keypoints = num_keypoints
        self.voxel_res = voxel_resolution
        
        # 1. 关键点编码器（3D关键点 → 特征）
        self.kp_encoder = nn.Sequential(
            nn.Linear(num_keypoints * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
        
        # 2. 特征压缩/扩展（降低计算成本）
        compressed_dim = feature_dim // 4
        self.compress = nn.Conv2d(feature_dim, compressed_dim, 1)
        self.expand = nn.Conv2d(compressed_dim, feature_dim, 1)
        
        # 3. 3D体素网格生成器
        # 输入：压缩特征 + 3D坐标
        self.voxel_mlp = nn.Sequential(
            nn.Linear(compressed_dim + 3, 128),  # 特征+3D坐标
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, compressed_dim)
        )
        
        # 4. 姿态编码器（用于体积渲染）
        self.pose_encoder = nn.Sequential(
            nn.Linear(3, 32),  # 欧拉角
            nn.ReLU(),
            nn.Linear(32, compressed_dim)
        )
        
        # 5. 体积渲染（简化版）
        self.render_mlp = nn.Sequential(
            nn.Linear(compressed_dim * 2, 64),  # 体素特征 + 姿态特征
            nn.ReLU(),
            nn.Linear(64, compressed_dim)
        )
        
    def _create_voxel_coords(self, device):
        """创建3D坐标网格"""
        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, self.voxel_res, device=device),
            torch.linspace(-1, 1, self.voxel_res, device=device),
            torch.linspace(-1, 1, self.voxel_res, device=device),
            indexing='ij'
        ), dim=-1)  # [res, res, res, 3]
        return coords.flatten(0, 2)  # [res^3, 3]
    
    def forward(self, features, keypoints_3d, target_pose, source_pose=None):
        """
        前向传播
        
        Args:
            features: [batch, C, H, W] 源视角特征（如果是1D特征，会重塑为2D）
            keypoints_3d: [batch, num_kp, 3] 3D关键点
            target_pose: [batch, 3] 目标姿态（欧拉角）
            source_pose: [batch, 3] 源姿态（可选，用于更精确的变换）
            
        Returns:
            output: [batch, C, H, W] 投影后的特征
        """
        batch = features.shape[0]
        device = features.device
        
        # 处理输入维度
        if len(features.shape) == 2:
            # [batch, feature_dim] -> [batch, feature_dim, 1, 1]
            features = features.unsqueeze(-1).unsqueeze(-1)
            is_1d = True
        else:
            is_1d = False
        
        _, C, H, W = features.shape
        
        # 1. 编码关键点为3D先验
        kp_flat = keypoints_3d.flatten(1)  # [batch, num_kp * 3]
        kp_features = self.kp_encoder(kp_flat)  # [batch, feature_dim]
        
        # 2. 压缩特征
        compressed = self.compress(features)  # [batch, C/4, H, W]
        compressed_dim = compressed.shape[1]
        
        # 3. 生成3D体素网格（简化版：基于关键点插值）
        # 创建3D坐标网格
        voxel_coords = self._create_voxel_coords(device)  # [res^3, 3]
        
        # 扩展批次维度
        voxel_coords = voxel_coords.unsqueeze(0).expand(batch, -1, -1)  # [batch, res^3, 3]
        
        # 重复关键点特征到每个体素
        kp_features_expanded = kp_features.unsqueeze(1).expand(-1, voxel_coords.shape[1], -1)  # [batch, res^3, feature_dim]
        
        # 使用压缩后的特征维度（降低计算成本）
        kp_features_compressed = F.adaptive_avg_pool2d(
            kp_features_expanded.view(batch, -1, 1, 1),
            (1, 1)
        ).squeeze(-1).squeeze(-1)  # [batch, res^3]
        
        # 简化为使用平均特征
        avg_compressed = compressed.mean(dim=[2, 3])  # [batch, C/4]
        avg_compressed_expanded = avg_compressed.unsqueeze(1).expand(-1, voxel_coords.shape[1], -1)  # [batch, res^3, C/4]
        
        # 生成体素特征（特征 + 3D坐标）
        voxel_input = torch.cat([avg_compressed_expanded, voxel_coords], dim=-1)  # [batch, res^3, C/4 + 3]
        voxel_features = self.voxel_mlp(voxel_input)  # [batch, res^3, C/4]
        
        # 4. 体积渲染到目标视角
        # 编码目标姿态
        pose_features = self.pose_encoder(target_pose)  # [batch, C/4]
        pose_features_expanded = pose_features.unsqueeze(1).expand(-1, voxel_features.shape[1], -1)  # [batch, res^3, C/4]
        
        # 融合体素特征和姿态特征
        render_input = torch.cat([voxel_features, pose_features_expanded], dim=-1)  # [batch, res^3, C/4 * 2]
        rendered_features = self.render_mlp(render_input)  # [batch, res^3, C/4]
        
        # 5. 聚合体素特征（平均池化）
        aggregated = rendered_features.mean(1)  # [batch, C/4]
        
        # 6. 扩展回原始特征维度并空间广播
        aggregated = aggregated.unsqueeze(-1).unsqueeze(-1)  # [batch, C/4, 1, 1]
        expanded = self.expand(aggregated)  # [batch, C, 1, 1]
        expanded = expanded.expand(-1, -1, H, W)  # [batch, C, H, W]
        
        # 7. 与原始特征融合（残差连接）
        output = features + 0.1 * expanded  # 小权重融合，保持原始特征主导
        
        # 如果输入是1D，输出也应该是1D
        if is_1d:
            output = output.squeeze(-1).squeeze(-1)  # [batch, C]
        
        return output

