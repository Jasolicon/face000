"""
基于SENet的三元组网络：使用Squeeze-and-Excitation机制控制身份特征和姿态特征

核心思想：
1. 使用SE Block学习哪些维度应该保留（身份特征）
2. 使用SE Block学习哪些维度应该转换（姿态特征）
3. 通过双分支架构明确分离身份和姿态特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from typing import Optional, Tuple

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from train_transformer3D.triplet.models_utils import set_seed

# 设置随机种子
set_seed(42)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    通过全局平均池化 + 全连接层生成通道权重
    
    参考：Squeeze-and-Excitation Networks (CVPR 2018)
    
    对于1D特征向量 [batch, channels]：
    - 方案1（推荐）：每个样本独立计算权重，更灵活
    - 方案2：使用batch统计，所有样本共享权重，计算量小
    """
    def __init__(self, channels: int, reduction: int = 16, use_batch_stat: bool = False):
        """
        Args:
            channels: 特征维度（通道数）
            reduction: 压缩比例，用于减少参数量
            use_batch_stat: 是否使用batch统计（False=每个样本独立，True=共享权重）
        """
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.use_batch_stat = use_batch_stat
        
        # Squeeze: 全局平均池化（对于1D特征，直接使用特征本身）
        # Excitation: 两个全连接层生成通道权重
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch, channels] 或 [batch, channels, H, W]
            
        Returns:
            output: 加权后的特征，形状与输入相同
        """
        batch_size = x.size(0)
        
        # Squeeze: 全局平均池化
        if x.dim() == 2:
            # 1D特征 [batch, channels]
            if self.use_batch_stat:
                # 方案2：使用batch统计（所有样本共享权重）
                se = x.mean(dim=0, keepdim=True)  # [1, channels]
                weights = self.fc(se)  # [1, channels] -> [1, channels]
                weights = weights.expand(batch_size, -1)  # [batch, channels]
            else:
                # 方案1（推荐）：每个样本独立计算权重
                # 对于1D特征，特征本身已经是全局统计，直接使用
                se = x  # [batch, channels]
                weights = self.fc(se)  # [batch, channels] -> [batch, channels]
        elif x.dim() == 4:
            # 2D特征 [batch, channels, H, W]
            se = F.adaptive_avg_pool2d(x, 1).view(batch_size, -1)  # [batch, channels]
            weights = self.fc(se)  # [batch, channels] -> [batch, channels]
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")
        
        # Scale: 应用权重
        if x.dim() == 2:
            return x * weights
        else:
            return x * weights.view(batch_size, x.size(1), 1, 1)


class DualBranchSENet(nn.Module):
    """
    双分支SENet：分离身份特征和姿态特征
    
    架构：
    1. 身份分支：保护高相似维度（如维度60, 312, 459等）
    2. 姿态分支：学习低相似维度转换（如维度229, 334, 437等）
    3. 融合：加权求和生成正面特征
    """
    def __init__(
        self,
        feature_dim: int = 512,
        reduction: int = 16,
        fusion_alpha: float = 0.7,  # 身份权重初始值
        learnable_fusion: bool = True,  # 是否学习融合权重
        use_batch_stat: bool = False  # SE Block是否使用batch统计（False=每个样本独立，推荐）
    ):
        """
        Args:
            feature_dim: 特征维度
            reduction: SE Block的压缩比例
            fusion_alpha: 身份分支的初始权重（0-1之间）
            learnable_fusion: 是否让融合权重可学习
            use_batch_stat: SE Block是否使用batch统计（False=每个样本独立，推荐）
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.reduction = reduction
        
        # ========== 身份分支：保护高相似维度 ==========
        self.identity_se = SEBlock(feature_dim, reduction, use_batch_stat=use_batch_stat)
        self.identity_fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # ========== 姿态分支：学习低相似维度转换 ==========
        self.pose_se = SEBlock(feature_dim, reduction, use_batch_stat=use_batch_stat)
        self.pose_fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )
        
        # ========== 融合权重 ==========
        if learnable_fusion:
            # 可学习的融合权重（通过sigmoid限制在[0,1]）
            self.fusion_alpha = nn.Parameter(torch.tensor(fusion_alpha))
        else:
            # 固定融合权重
            self.register_buffer('fusion_alpha', torch.tensor(fusion_alpha))
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_branches: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch, feature_dim]
            return_branches: 是否返回分支输出（用于分析）
            
        Returns:
            front_features: 融合后的正面特征 [batch, feature_dim]
            identity_features: 身份分支输出 [batch, feature_dim]（如果return_branches=True）
            pose_features: 姿态分支输出 [batch, feature_dim]（如果return_branches=True）
        """
        # ========== 身份分支：保留原始特征 ==========
        identity_se_out = self.identity_se(x)  # SE Block加权
        identity_features = self.identity_fc(identity_se_out)  # 全连接层
        
        # 残差连接：保护原始特征
        identity_features = identity_features + x  # [batch, feature_dim]
        
        # ========== 姿态分支：学习转换 ==========
        pose_se_out = self.pose_se(x)  # SE Block加权
        pose_features = self.pose_fc(pose_se_out)  # 全连接层
        
        # ========== 融合：加权求和 ==========
        if isinstance(self.fusion_alpha, nn.Parameter):
            # 可学习权重，通过sigmoid限制在[0,1]
            alpha = torch.sigmoid(self.fusion_alpha)
        else:
            # 固定权重
            alpha = self.fusion_alpha
        
        front_features = alpha * identity_features + (1 - alpha) * pose_features
        
        if return_branches:
            return front_features, identity_features, pose_features
        else:
            return front_features, None, None


class SENetTripletNetwork(nn.Module):
    """
    基于SENet的三元组网络
    
    架构：
    1. 图像特征（src） + 姿势特征（pose） -> 连接 -> 融合特征
    2. 双分支SENet -> 正面特征（front_features）
    3. 身份投影头 -> 身份特征（identity_features）
    """
    def __init__(
        self,
        image_dim: int = 512,  # 图像特征维度（InsightFace: 512）
        pose_dim: int = 3,    # 姿势维度（欧拉角: 3）
        hidden_dim: int = 1024,  # 隐藏层维度
        num_layers: int = 3,  # 全连接层数量
        dropout: float = 0.1,
        activation: str = 'relu',
        se_reduction: int = 16,  # SE Block压缩比例
        fusion_alpha: float = 0.7,  # 身份分支初始权重
        learnable_fusion: bool = True,  # 是否学习融合权重
        use_batch_stat: bool = False,  # SE Block是否使用batch统计（False=每个样本独立，推荐）
        identity_dim: int = 512  # 身份特征维度
    ):
        """
        初始化SENet三元组网络
        
        Args:
            image_dim: 图像特征维度
            pose_dim: 姿势维度
            hidden_dim: 隐藏层维度
            num_layers: 全连接层数量
            dropout: Dropout比率
            activation: 激活函数类型
            se_reduction: SE Block压缩比例
            fusion_alpha: 身份分支初始权重
            learnable_fusion: 是否学习融合权重
            use_batch_stat: SE Block是否使用batch统计（False=每个样本独立，推荐）
            identity_dim: 身份特征维度
        """
        super().__init__()
        
        self.image_dim = image_dim
        self.pose_dim = pose_dim
        self.hidden_dim = hidden_dim
        self.identity_dim = identity_dim
        
        # 输入维度：图像特征 + 姿势特征
        input_dim = image_dim + pose_dim
        
        # ========== 特征融合层 ==========
        # 将图像特征和姿势特征融合
        self.fusion_layers = nn.ModuleList()
        
        # 第一层：融合输入
        self.fusion_layers.append(
            self._make_fc_block(input_dim, hidden_dim, dropout, activation)
        )
        
        # 中间层
        for _ in range(num_layers - 2):
            self.fusion_layers.append(
                self._make_fc_block(hidden_dim, hidden_dim, dropout, activation)
            )
        
        # ========== 双分支SENet ==========
        self.dual_branch = DualBranchSENet(
            feature_dim=image_dim,  # 注意：这里是image_dim，不是hidden_dim
            reduction=se_reduction,
            fusion_alpha=fusion_alpha,
            learnable_fusion=learnable_fusion,
            use_batch_stat=use_batch_stat
        )
        
        # ========== 正面特征生成层 ==========
        # 从融合特征生成图像特征维度的特征
        self.front_feature_projection = nn.Sequential(
            nn.Linear(hidden_dim, image_dim),
            nn.BatchNorm1d(image_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # ========== 正面姿势生成层 ==========
        self.front_pose_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.front_pose_layers.append(
                self._make_fc_block(hidden_dim, hidden_dim // 2, dropout, activation)
            )
        self.front_pose_output = nn.Linear(hidden_dim // 2, pose_dim)
        
        # ========== 身份投影头（用于三元组损失） ==========
        self.identity_head = nn.Sequential(
            nn.Linear(image_dim, image_dim),
            nn.BatchNorm1d(image_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(image_dim, identity_dim),
            nn.BatchNorm1d(identity_dim)
        )
        
        self._init_weights()
    
    def _make_fc_block(self, in_dim, out_dim, dropout, activation):
        """创建全连接块"""
        layers = [
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        ]
        
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'gelu':
            layers.append(nn.GELU())
        else:
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Dropout(dropout))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        src: torch.Tensor,  # 图像特征 [batch, image_dim]
        pose: torch.Tensor,  # 姿势特征 [batch, pose_dim]
        return_identity_features: bool = True,  # 是否返回身份特征（用于三元组损失）
        return_front_pose: bool = True,  # 是否返回正面姿势
        return_branches: bool = False  # 是否返回SENet分支输出（用于分析）
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            src: 图像特征 [batch, image_dim]
            pose: 姿势特征 [batch, pose_dim]
            return_identity_features: 是否返回身份特征（用于三元组损失）
            return_front_pose: 是否返回正面姿势
            return_branches: 是否返回SENet分支输出（用于分析）
            
        Returns:
            front_features: 正面图像特征 [batch, image_dim]
            identity_features: 归一化的身份特征 [batch, identity_dim]（如果return_identity_features=True）
            front_pose: 正面姿势 [batch, pose_dim]（如果return_front_pose=True）
        """
        batch_size = src.size(0)
        
        # ========== 阶段1：特征融合 ==========
        # 连接图像特征和姿势特征
        combined = torch.cat([src, pose], dim=1)  # [batch, image_dim + pose_dim]
        
        # 通过融合层
        fused_features = combined
        for layer in self.fusion_layers:
            fused_features = layer(fused_features)  # [batch, hidden_dim]
        
        # ========== 阶段2：生成正面特征（使用双分支SENet） ==========
        # 将融合特征投影到图像特征维度
        projected_features = self.front_feature_projection(fused_features)  # [batch, image_dim]
        
        # 使用双分支SENet生成正面特征
        front_features, identity_branch, pose_branch = self.dual_branch(
            projected_features,
            return_branches=return_branches
        )  # [batch, image_dim]
        
        # 残差连接：正面特征 = 原始特征 + SENet输出
        front_features = src + front_features  # [batch, image_dim]
        
        # ========== 阶段3：生成正面姿势 ==========
        front_pose = None
        if return_front_pose:
            # 通过正面姿势生成层
            front_pose_hidden = fused_features
            for layer in self.front_pose_layers:
                front_pose_hidden = layer(front_pose_hidden)  # [batch, hidden_dim // 2]
            
            # 生成正面姿势
            front_pose = self.front_pose_output(front_pose_hidden)  # [batch, pose_dim]
        
        # ========== 阶段4：生成身份特征（用于三元组损失） ==========
        identity_features = None
        if return_identity_features:
            # 使用正面特征生成身份特征
            identity_features = self.identity_head(front_features)  # [batch, identity_dim]
            
            # L2归一化（用于余弦相似度计算）
            identity_features = F.normalize(identity_features, p=2, dim=1)
        
        if return_branches:
            # 返回分支输出用于分析
            return front_features, identity_features, front_pose, identity_branch, pose_branch
        else:
            return front_features, identity_features, front_pose
    
    def get_trainable_parameters(self):
        """获取可训练参数"""
        return list(self.parameters())
    
    def get_fusion_alpha(self):
        """获取当前融合权重（用于监控）"""
        if isinstance(self.dual_branch.fusion_alpha, nn.Parameter):
            return torch.sigmoid(self.dual_branch.fusion_alpha).item()
        else:
            return self.dual_branch.fusion_alpha.item()


if __name__ == "__main__":
    """测试代码"""
    print("=" * 70)
    print("SENet三元组网络测试")
    print("=" * 70)
    
    # 创建模型
    model = SENetTripletNetwork(
        image_dim=512,
        pose_dim=3,
        hidden_dim=1024,
        num_layers=3,
        se_reduction=16,
        fusion_alpha=0.7,
        learnable_fusion=True
    )
    
    # 测试前向传播
    batch_size = 4
    src = torch.randn(batch_size, 512)
    pose = torch.randn(batch_size, 3)
    
    print(f"\n输入形状:")
    print(f"  图像特征: {src.shape}")
    print(f"  姿势特征: {pose.shape}")
    
    # 前向传播
    front_features, identity_features, front_pose = model(
        src=src,
        pose=pose,
        return_identity_features=True,
        return_front_pose=True,
        return_branches=False
    )
    
    print(f"\n输出形状:")
    print(f"  正面特征: {front_features.shape}")
    print(f"  身份特征: {identity_features.shape}")
    print(f"  正面姿势: {front_pose.shape}")
    
    # 测试分支输出
    front_features, identity_features, front_pose, identity_branch, pose_branch = model(
        src=src,
        pose=pose,
        return_identity_features=True,
        return_front_pose=True,
        return_branches=True
    )
    
    print(f"\n分支输出形状:")
    print(f"  身份分支: {identity_branch.shape}")
    print(f"  姿态分支: {pose_branch.shape}")
    
    # 测试融合权重
    fusion_alpha = model.get_fusion_alpha()
    print(f"\n当前融合权重 (α): {fusion_alpha:.4f}")
    print(f"  身份分支权重: {fusion_alpha:.4f}")
    print(f"  姿态分支权重: {1 - fusion_alpha:.4f}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数量:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)

