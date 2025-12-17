"""
模型1：特征角度转换ControlNet
将姿势信息和特征信息输入，受另一个角度调控，获得该角度的特征，保持身份一致性

架构设计：
- 基础网络：特征编码器（冻结或可训练）
- ControlNet：角度条件控制网络（零卷积初始化）
- 输出：目标角度的特征（保持身份）
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
from train_transformer3D.models_utils import AnglePositionalEncoding

set_seed(42)


class ZeroConv1d(nn.Module):
    """
    零卷积层（Zero Convolution）
    ControlNet的核心：初始化为零，确保训练初期不影响基础网络
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)
        # 初始化为零
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, in_channels, length] 或 [batch, in_channels]
        Returns:
            output: [batch, out_channels, length] 或 [batch, out_channels]
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [batch, in_channels] -> [batch, in_channels, 1]
        out = self.conv(x)
        if out.shape[-1] == 1:
            out = out.squeeze(-1)  # [batch, out_channels, 1] -> [batch, out_channels]
        return out


class ZeroLinear(nn.Module):
    """
    零线性层（Zero Linear）
    用于全连接层的零初始化
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # 初始化为零
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class AngleControlBlock(nn.Module):
    """
    角度控制块：将角度条件编码为控制特征
    """
    def __init__(self, pose_dim: int = 3, feature_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.pose_dim = pose_dim
        self.feature_dim = feature_dim
        
        # 角度编码器
        self.angle_encoder = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # 角度位置编码（可选）
        self.angle_pe = AnglePositionalEncoding(feature_dim, angle_dim=pose_dim)
        
    def forward(self, target_pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target_pose: 目标姿态 [batch, pose_dim]
        Returns:
            control_features: 控制特征 [batch, feature_dim]
        """
        # 方法1：直接编码
        encoded = self.angle_encoder(target_pose)  # [batch, feature_dim]
        
        # 方法2：位置编码（增强角度信息）
        pe = self.angle_pe(target_pose)  # [batch, feature_dim]
        
        # 融合
        control_features = encoded + 0.1 * pe
        
        return control_features


class FeatureEncoder(nn.Module):
    """
    特征编码器：将输入特征编码为隐藏表示
    """
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 512, num_layers: int = 2):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        layers = []
        for i in range(num_layers):
            in_dim = feature_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 输入特征 [batch, feature_dim]
        Returns:
            encoded: 编码后的特征 [batch, hidden_dim]
        """
        return self.encoder(features)


class ControlNetBlock(nn.Module):
    """
    ControlNet块：处理控制条件并生成控制信号
    """
    def __init__(self, feature_dim: int = 512, control_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim
        self.control_dim = control_dim
        
        # 控制条件处理
        self.control_proj = nn.Sequential(
            nn.Linear(control_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # 特征和控制融合
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # 零卷积输出（ControlNet关键）
        self.zero_conv = ZeroLinear(feature_dim, feature_dim)
    
    def forward(self, features: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 基础特征 [batch, feature_dim]
            control: 控制条件 [batch, control_dim]
        Returns:
            control_signal: 控制信号 [batch, feature_dim]
        """
        # 处理控制条件
        control_proj = self.control_proj(control)  # [batch, feature_dim]
        
        # 融合特征和控制
        fused = torch.cat([features, control_proj], dim=-1)  # [batch, feature_dim * 2]
        fused = self.fusion(fused)  # [batch, feature_dim]
        
        # 零卷积输出（初始时输出为零，不影响基础网络）
        control_signal = self.zero_conv(fused)  # [batch, feature_dim]
        
        return control_signal


class FeatureAngleControlNet(nn.Module):
    """
    模型1：特征角度转换ControlNet
    
    输入：
    - source_features: 源特征 [batch, feature_dim]
    - source_pose: 源姿态 [batch, pose_dim]
    - target_pose: 目标姿态（控制条件）[batch, pose_dim]
    
    输出：
    - target_features: 目标角度的特征 [batch, feature_dim]
    
    设计要点：
    1. 基础网络处理源特征和源姿态
    2. ControlNet根据目标姿态生成控制信号
    3. 通过残差连接融合，保持身份一致性
    """
    def __init__(
        self,
        feature_dim: int = 512,
        pose_dim: int = 3,
        hidden_dim: int = 512,
        num_encoder_layers: int = 2,
        num_control_layers: int = 3,
        freeze_base: bool = False,  # 是否冻结基础网络
        identity_weight: float = 0.8  # 身份保持权重
    ):
        """
        Args:
            feature_dim: 特征维度
            pose_dim: 姿态维度
            hidden_dim: 隐藏层维度
            num_encoder_layers: 编码器层数
            num_control_layers: ControlNet层数
            freeze_base: 是否冻结基础网络（ControlNet标准做法）
            identity_weight: 身份保持权重（0-1，越大越保持身份）
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.pose_dim = pose_dim
        self.hidden_dim = hidden_dim
        self.identity_weight = identity_weight
        
        # ========== 基础网络：特征编码器 ==========
        self.feature_encoder = FeatureEncoder(feature_dim, hidden_dim, num_encoder_layers)
        
        # 源姿态编码器
        self.source_pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 基础特征融合
        self.base_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # 冻结基础网络（可选）
        if freeze_base:
            for param in self.feature_encoder.parameters():
                param.requires_grad = False
            for param in self.source_pose_encoder.parameters():
                param.requires_grad = False
        
        # ========== ControlNet：角度条件控制网络 ==========
        # 角度控制编码器
        self.angle_control = AngleControlBlock(pose_dim, feature_dim, hidden_dim)
        
        # ControlNet块
        self.control_blocks = nn.ModuleList([
            ControlNetBlock(hidden_dim, feature_dim)
            for _ in range(num_control_layers)
        ])
        
        # ========== 输出层 ==========
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feature_dim)
        )
        
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
        source_features: torch.Tensor,
        source_pose: torch.Tensor,
        target_pose: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            source_features: 源特征 [batch, feature_dim]
            source_pose: 源姿态 [batch, pose_dim]
            target_pose: 目标姿态（控制条件）[batch, pose_dim]
        
        Returns:
            target_features: 目标角度的特征 [batch, feature_dim]
        """
        batch_size = source_features.size(0)
        
        # ========== 阶段1：基础网络处理 ==========
        # 编码源特征
        encoded_features = self.feature_encoder(source_features)  # [batch, hidden_dim]
        
        # 编码源姿态
        encoded_pose = self.source_pose_encoder(source_pose)  # [batch, hidden_dim]
        
        # 融合源特征和源姿态
        base_features = torch.cat([encoded_features, encoded_pose], dim=-1)  # [batch, hidden_dim * 2]
        base_features = self.base_fusion(base_features)  # [batch, hidden_dim]
        
        # ========== 阶段2：ControlNet生成控制信号 ==========
        # 角度控制编码
        control_condition = self.angle_control(target_pose)  # [batch, feature_dim]
        
        # 通过ControlNet块生成控制信号
        control_signal = base_features
        for control_block in self.control_blocks:
            control_signal = control_block(control_signal, control_condition)  # [batch, hidden_dim]
            # 残差连接
            control_signal = control_signal + base_features
        
        # ========== 阶段3：融合并输出 ==========
        # 融合基础特征和控制信号
        # 使用可学习的权重平衡身份保持和角度转换
        fused_features = self.identity_weight * base_features + (1 - self.identity_weight) * control_signal
        
        # 输出投影
        target_features = self.output_proj(fused_features)  # [batch, feature_dim]
        
        # 残差连接：保持身份一致性
        target_features = source_features + target_features
        
        return target_features
    
    def get_trainable_parameters(self):
        """获取可训练参数（用于优化器配置）"""
        return list(self.parameters())


if __name__ == "__main__":
    """测试代码"""
    print("=" * 70)
    print("特征角度转换ControlNet测试")
    print("=" * 70)
    
    # 创建模型
    model = FeatureAngleControlNet(
        feature_dim=512,
        pose_dim=3,
        hidden_dim=512,
        num_encoder_layers=2,
        num_control_layers=3,
        freeze_base=False,
        identity_weight=0.8
    )
    
    # 测试前向传播
    batch_size = 4
    source_features = torch.randn(batch_size, 512)
    source_pose = torch.randn(batch_size, 3)
    target_pose = torch.randn(batch_size, 3)
    
    print(f"\n输入形状:")
    print(f"  源特征: {source_features.shape}")
    print(f"  源姿态: {source_pose.shape}")
    print(f"  目标姿态: {target_pose.shape}")
    
    # 前向传播
    target_features = model(source_features, source_pose, target_pose)
    
    print(f"\n输出形状:")
    print(f"  目标特征: {target_features.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数量:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)

