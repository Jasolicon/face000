"""
模型1：FeatureAngleControlNet（特征角度控制）
将姿势信息和特征信息输入，受另一个角度调控，获得该角度的特征，保持身份一致性

参考 ControlNet 架构：
- 零卷积初始化：确保训练初期不影响主网络
- 条件注入：通过残差连接注入角度控制信号
- 身份保持：通过身份一致性约束确保不同角度的特征属于同一身份
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from typing import Optional, Tuple

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from train_transformer3D.utils_seed import set_seed
from train_transformer3D.controlnetlike.utils import ZeroLinear, compute_identity_consistency_loss

# 设置随机种子
set_seed(42)


class AngleControlBlock(nn.Module):
    """
    角度控制块（ControlNet-like）
    处理目标角度，生成控制信号
    """
    def __init__(self, pose_dim=3, hidden_dim=256, num_layers=3):
        super().__init__()
        self.pose_dim = pose_dim
        self.hidden_dim = hidden_dim
        
        # 角度编码器
        layers = []
        in_dim = pose_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.1))
            in_dim = hidden_dim
        
        self.angle_encoder = nn.Sequential(*layers)
        
        # 零卷积输出（ControlNet核心）
        self.zero_conv = ZeroLinear(hidden_dim, hidden_dim)
    
    def forward(self, target_angle):
        """
        Args:
            target_angle: 目标角度 [batch, pose_dim]
        Returns:
            control_signal: 控制信号 [batch, hidden_dim]
        """
        encoded = self.angle_encoder(target_angle)
        control = self.zero_conv(encoded)
        return control


class IdentityPreservationModule(nn.Module):
    """
    身份保持模块
    提取角度无关的身份特征，确保身份一致性
    """
    def __init__(self, feature_dim=512, identity_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        self.identity_dim = identity_dim
        
        # 身份特征提取器
        self.identity_extractor = nn.Sequential(
            nn.Linear(feature_dim, identity_dim),
            nn.LayerNorm(identity_dim),
            nn.ReLU(inplace=True),
            nn.Linear(identity_dim, identity_dim),
            nn.LayerNorm(identity_dim)
        )
        
        # 身份特征投影回原空间
        self.identity_projector = nn.Sequential(
            nn.Linear(identity_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
    
    def forward(self, features):
        """
        Args:
            features: 输入特征 [batch, feature_dim]
        Returns:
            identity_features: 身份特征 [batch, identity_dim]
            projected_features: 投影回原空间的特征 [batch, feature_dim]
        """
        identity_features = self.identity_extractor(features)
        projected_features = self.identity_projector(identity_features)
        return identity_features, projected_features


class FeatureAngleControlNet(nn.Module):
    """
    特征角度控制网络
    
    输入：
    - src_features: 源特征 [batch, feature_dim]
    - src_pose: 源姿势 [batch, pose_dim]
    - target_angle: 目标角度 [batch, pose_dim]（控制信号）
    
    输出：
    - target_features: 目标角度特征 [batch, feature_dim]（保持身份一致性）
    """
    def __init__(
        self,
        feature_dim: int = 512,
        pose_dim: int = 3,
        hidden_dim: int = 1024,
        identity_dim: int = 256,
        num_control_layers: int = 3,
        num_fusion_layers: int = 3,
        dropout: float = 0.1
    ):
        """
        Args:
            feature_dim: 特征维度（默认512，InsightFace）
            pose_dim: 姿势维度（默认3，欧拉角）
            hidden_dim: 隐藏层维度
            identity_dim: 身份特征维度
            num_control_layers: 控制网络层数
            num_fusion_layers: 融合网络层数
            dropout: Dropout比率
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.pose_dim = pose_dim
        self.hidden_dim = hidden_dim
        self.identity_dim = identity_dim
        
        # ========== 1. 特征编码器 ==========
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # ========== 2. 姿势编码器 ==========
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # ========== 3. 角度控制网络（ControlNet-like） ==========
        self.angle_control = AngleControlBlock(
            pose_dim=pose_dim,
            hidden_dim=hidden_dim,
            num_layers=num_control_layers
        )
        
        # ========== 4. 身份保持模块 ==========
        self.identity_module = IdentityPreservationModule(
            feature_dim=feature_dim,
            identity_dim=identity_dim
        )
        
        # ========== 5. 特征融合网络 ==========
        # 输入：特征编码 + 姿势编码 + 控制信号
        fusion_input_dim = hidden_dim * 3  # 特征 + 姿势 + 控制
        
        fusion_layers = []
        in_dim = fusion_input_dim
        for i in range(num_fusion_layers):
            fusion_layers.append(nn.Linear(in_dim, hidden_dim))
            fusion_layers.append(nn.LayerNorm(hidden_dim))
            fusion_layers.append(nn.ReLU(inplace=True))
            fusion_layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        self.fusion_network = nn.Sequential(*fusion_layers)
        
        # ========== 6. 输出投影层 ==========
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # ========== 7. 零卷积输出（ControlNet核心） ==========
        self.zero_conv_output = ZeroLinear(feature_dim, feature_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        src_features: torch.Tensor,
        src_pose: torch.Tensor,
        target_angle: torch.Tensor,
        return_identity: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            src_features: 源特征 [batch, feature_dim]
            src_pose: 源姿势 [batch, pose_dim]
            target_angle: 目标角度 [batch, pose_dim]（控制信号）
            return_identity: 是否返回身份特征
        
        Returns:
            target_features: 目标角度特征 [batch, feature_dim]
            identity_features: 身份特征 [batch, identity_dim]（如果return_identity=True）
        """
        batch_size = src_features.size(0)
        
        # ========== 1. 编码输入 ==========
        # 特征编码
        feature_encoded = self.feature_encoder(src_features)  # [batch, hidden_dim]
        
        # 姿势编码
        pose_encoded = self.pose_encoder(src_pose)  # [batch, hidden_dim]
        
        # ========== 2. 角度控制网络（ControlNet-like） ==========
        control_signal = self.angle_control(target_angle)  # [batch, hidden_dim]
        
        # ========== 3. 身份保持 ==========
        identity_features, identity_projected = self.identity_module(src_features)
        # identity_projected: [batch, feature_dim] - 身份相关的特征
        
        # ========== 4. 特征融合 ==========
        # 拼接所有编码特征
        fused = torch.cat([
            feature_encoded,    # [batch, hidden_dim]
            pose_encoded,       # [batch, hidden_dim]
            control_signal      # [batch, hidden_dim]
        ], dim=1)  # [batch, hidden_dim * 3]
        
        # 通过融合网络
        fused_features = self.fusion_network(fused)  # [batch, hidden_dim]
        
        # ========== 5. 生成目标特征 ==========
        # 投影到特征维度
        target_features_raw = self.output_projection(fused_features)  # [batch, feature_dim]
        
        # 零卷积输出（ControlNet核心：训练初期不影响主网络）
        target_features_delta = self.zero_conv_output(target_features_raw)  # [batch, feature_dim]
        
        # 残差连接：原始特征 + 身份特征 + 角度转换
        target_features = src_features + 0.1 * identity_projected + target_features_delta
        
        if return_identity:
            return target_features, identity_features
        else:
            return target_features, None
    
    def compute_identity_loss(
        self,
        src_features: torch.Tensor,
        src_pose: torch.Tensor,
        target_angles: torch.Tensor,
        num_samples: int = 3
    ) -> torch.Tensor:
        """
        计算身份一致性损失
        通过生成多个角度的特征，确保它们属于同一身份
        
        Args:
            src_features: 源特征 [batch, feature_dim]
            src_pose: 源姿势 [batch, pose_dim]
            target_angles: 目标角度列表 [num_samples, batch, pose_dim] 或单个 [batch, pose_dim]
            num_samples: 如果target_angles是单个，则生成num_samples个随机角度
        
        Returns:
            identity_loss: 身份一致性损失
        """
        if target_angles.dim() == 2:
            # 单个目标角度，生成多个随机角度
            batch_size = target_angles.size(0)
            device = target_angles.device
            
            # 生成随机角度（在合理范围内）
            random_angles = []
            for _ in range(num_samples):
                angle = torch.randn(batch_size, self.pose_dim, device=device) * 0.5
                random_angles.append(angle)
            target_angles = torch.stack(random_angles, dim=0)  # [num_samples, batch, pose_dim]
        
        # 生成多个角度的特征
        features_list = []
        for i in range(target_angles.size(0)):
            angle = target_angles[i]  # [batch, pose_dim]
            target_feat, identity_feat = self.forward(
                src_features, src_pose, angle, return_identity=True
            )
            features_list.append(identity_feat)
        
        # 计算身份一致性损失
        identity_loss = compute_identity_consistency_loss(features_list)
        
        return identity_loss


if __name__ == "__main__":
    """测试代码"""
    print("=" * 70)
    print("FeatureAngleControlNet 测试")
    print("=" * 70)
    
    # 创建模型
    model = FeatureAngleControlNet(
        feature_dim=512,
        pose_dim=3,
        hidden_dim=1024,
        identity_dim=256
    )
    
    # 测试前向传播
    batch_size = 4
    src_features = torch.randn(batch_size, 512)
    src_pose = torch.randn(batch_size, 3)
    target_angle = torch.randn(batch_size, 3)
    
    print(f"\n输入形状:")
    print(f"  源特征: {src_features.shape}")
    print(f"  源姿势: {src_pose.shape}")
    print(f"  目标角度: {target_angle.shape}")
    
    # 前向传播
    target_features, identity_features = model(
        src_features, src_pose, target_angle, return_identity=True
    )
    
    print(f"\n输出形状:")
    print(f"  目标特征: {target_features.shape}")
    print(f"  身份特征: {identity_features.shape}")
    
    # 测试身份一致性损失
    identity_loss = model.compute_identity_loss(
        src_features, src_pose, target_angle, num_samples=3
    )
    print(f"\n身份一致性损失: {identity_loss.item():.4f}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数量:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)

