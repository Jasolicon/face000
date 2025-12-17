"""
模型1：特征转换ControlNet
输入：姿势信息 + 特征信息
控制：另一个角度
输出：该角度的特征（保持身份一致性）

参考ControlNet设计：
- 主网络：冻结的预训练特征转换网络
- 控制网络：可训练的网络，接收控制角度
- 零卷积：将控制信号注入主网络
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
from train_transformer3D.models_utils import AnglePositionalEncoding, AngleConditionedLayerNorm

set_seed(42)


class ZeroConv1d(nn.Module):
    """
    零卷积层（Zero Convolution）
    ControlNet的核心：初始时输出为零，不干扰主网络
    训练过程中逐渐学习控制信号
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        # 初始化为零
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [batch, in_channels, length] 或 [batch, in_channels]
        Returns:
            output: 输出特征 [batch, out_channels, length] 或 [batch, out_channels]
        """
        if x.dim() == 2:
            # [batch, in_channels] -> [batch, in_channels, 1]
            x = x.unsqueeze(-1)
            output = self.conv(x)  # [batch, out_channels, 1]
            output = output.squeeze(-1)  # [batch, out_channels]
        else:
            output = self.conv(x)
        return output


class ControlBranch(nn.Module):
    """
    控制分支：接收控制角度，生成控制信号
    """
    def __init__(
        self,
        control_angle_dim: int = 3,  # 控制角度维度（欧拉角：3）
        feature_dim: int = 512,      # 特征维度
        hidden_dim: int = 256,       # 隐藏层维度
        num_layers: int = 3          # 层数
    ):
        super().__init__()
        self.control_angle_dim = control_angle_dim
        self.feature_dim = feature_dim
        
        # 角度编码器
        self.angle_encoder = nn.Sequential(
            nn.Linear(control_angle_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 控制信号生成层
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
        self.control_layers = nn.Sequential(*layers)
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, feature_dim)
        
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
    
    def forward(self, control_angle: torch.Tensor) -> torch.Tensor:
        """
        Args:
            control_angle: 控制角度 [batch, control_angle_dim]
        Returns:
            control_signal: 控制信号 [batch, feature_dim]
        """
        # 编码角度
        angle_features = self.angle_encoder(control_angle)  # [batch, hidden_dim]
        
        # 生成控制信号
        control_hidden = self.control_layers(angle_features)  # [batch, hidden_dim]
        control_signal = self.output_proj(control_hidden)  # [batch, feature_dim]
        
        return control_signal


class MainNetwork(nn.Module):
    """
    主网络：特征转换网络（可以冻结或微调）
    将输入特征和姿势转换为目标角度的特征
    """
    def __init__(
        self,
        feature_dim: int = 512,
        pose_dim: int = 3,
        hidden_dim: int = 512,
        num_layers: int = 3,
        freeze: bool = False  # 是否冻结主网络
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.pose_dim = pose_dim
        
        # 输入投影
        self.input_proj = nn.Linear(feature_dim + pose_dim, hidden_dim)
        
        # Transformer层（简化版，使用MLP代替）
        self.transformer_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
            for _ in range(num_layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, feature_dim)
        
        # 角度位置编码
        self.angle_pe = AnglePositionalEncoding(hidden_dim, pose_dim)
        
        # 角度条件归一化
        self.angle_norm = AngleConditionedLayerNorm(hidden_dim, pose_dim)
        
        if freeze:
            # 冻结主网络参数
            for param in self.parameters():
                param.requires_grad = False
        
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
        features: torch.Tensor,
        pose: torch.Tensor,
        target_angle: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: 输入特征 [batch, feature_dim]
            pose: 输入姿势 [batch, pose_dim]
            target_angle: 目标角度 [batch, pose_dim]
        Returns:
            output_features: 目标角度的特征 [batch, feature_dim]
        """
        # 拼接特征和姿势
        combined = torch.cat([features, pose], dim=1)  # [batch, feature_dim + pose_dim]
        
        # 输入投影
        x = self.input_proj(combined)  # [batch, hidden_dim]
        
        # 添加角度位置编码
        angle_pe = self.angle_pe(target_angle)  # [batch, hidden_dim]
        x = x + angle_pe
        
        # Transformer层处理
        for layer in self.transformer_layers:
            residual = x
            x = layer(x)
            x = x + residual  # 残差连接
        
        # 角度条件归一化
        x = self.angle_norm(x, target_angle)  # [batch, hidden_dim]
        
        # 输出投影
        output_features = self.output_proj(x)  # [batch, feature_dim]
        
        return output_features


class FeatureControlNet(nn.Module):
    """
    特征转换ControlNet
    
    架构：
    1. 主网络：接收（特征，姿势）-> 输出目标角度特征
    2. 控制分支：接收控制角度 -> 生成控制信号
    3. 零卷积：将控制信号注入主网络
    4. 输出：目标角度的特征（保持身份一致性）
    """
    def __init__(
        self,
        feature_dim: int = 512,
        pose_dim: int = 3,
        hidden_dim: int = 512,
        num_main_layers: int = 3,
        num_control_layers: int = 3,
        freeze_main: bool = False  # 是否冻结主网络
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.pose_dim = pose_dim
        
        # 主网络（特征转换网络）
        self.main_network = MainNetwork(
            feature_dim=feature_dim,
            pose_dim=pose_dim,
            hidden_dim=hidden_dim,
            num_layers=num_main_layers,
            freeze=freeze_main
        )
        
        # 控制分支
        self.control_branch = ControlBranch(
            control_angle_dim=pose_dim,
            feature_dim=feature_dim,
            hidden_dim=256,
            num_layers=num_control_layers
        )
        
        # 零卷积（将控制信号注入主网络）
        self.zero_conv = ZeroConv1d(feature_dim, feature_dim)
        
        # 身份保护层（确保身份一致性）
        self.identity_protection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
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
        features: torch.Tensor,      # 输入特征 [batch, feature_dim]
        pose: torch.Tensor,          # 输入姿势 [batch, pose_dim]
        control_angle: torch.Tensor, # 控制角度 [batch, pose_dim]
        return_control_signal: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            features: 输入特征 [batch, feature_dim]
            pose: 输入姿势 [batch, pose_dim]
            control_angle: 控制角度（目标角度）[batch, pose_dim]
            return_control_signal: 是否返回控制信号
        
        Returns:
            output_features: 目标角度的特征 [batch, feature_dim]
            control_signal: 控制信号 [batch, feature_dim]（如果return_control_signal=True）
        """
        # ========== 1. 主网络处理 ==========
        main_output = self.main_network(features, pose, control_angle)  # [batch, feature_dim]
        
        # ========== 2. 控制分支生成控制信号 ==========
        control_signal = self.control_branch(control_angle)  # [batch, feature_dim]
        
        # ========== 3. 零卷积处理控制信号 ==========
        # ZeroConv1d的forward方法已经处理了2D输入（会自动unsqueeze和squeeze）
        control_output = self.zero_conv(control_signal)  # [batch, feature_dim]
        
        # ========== 4. 融合主网络输出和控制信号 ==========
        # 控制信号通过零卷积后，初始时为零，不会干扰主网络
        # 训练过程中逐渐学习如何控制
        fused_features = main_output + control_output  # [batch, feature_dim]
        
        # ========== 5. 身份保护 ==========
        # 确保输出特征保持身份一致性
        identity_features = self.identity_protection(features)  # [batch, feature_dim]
        
        # 融合身份特征（保护身份信息）
        output_features = 0.7 * fused_features + 0.3 * identity_features  # [batch, feature_dim]
        
        if return_control_signal:
            return output_features, control_signal
        else:
            return output_features, None
    
    def get_trainable_parameters(self):
        """获取可训练参数"""
        if self.main_network.parameters().__next__().requires_grad:
            # 主网络可训练
            return list(self.parameters())
        else:
            # 只返回控制分支和零卷积的参数
            trainable_params = []
            trainable_params.extend(list(self.control_branch.parameters()))
            trainable_params.extend(list(self.zero_conv.parameters()))
            trainable_params.extend(list(self.identity_protection.parameters()))
            return trainable_params


if __name__ == "__main__":
    """测试代码"""
    print("=" * 70)
    print("特征转换ControlNet测试")
    print("=" * 70)
    
    # 创建模型
    model = FeatureControlNet(
        feature_dim=512,
        pose_dim=3,
        hidden_dim=512,
        num_main_layers=3,
        num_control_layers=3,
        freeze_main=False
    )
    
    # 测试前向传播
    batch_size = 4
    features = torch.randn(batch_size, 512)
    pose = torch.randn(batch_size, 3)
    control_angle = torch.randn(batch_size, 3)
    
    print(f"\n输入形状:")
    print(f"  特征: {features.shape}")
    print(f"  姿势: {pose.shape}")
    print(f"  控制角度: {control_angle.shape}")
    
    # 前向传播
    output_features, control_signal = model(
        features=features,
        pose=pose,
        control_angle=control_angle,
        return_control_signal=True
    )
    
    print(f"\n输出形状:")
    print(f"  输出特征: {output_features.shape}")
    print(f"  控制信号: {control_signal.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数量:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
