"""
简单三元组网络：使用图像特征+姿势特征生成正面特征+正面姿势
使用带残差的全连接层，代替复杂的Transformer矫正网络
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


class ResidualFCBlock(nn.Module):
    """
    带残差连接的全连接块
    """
    def __init__(self, in_dim, out_dim, dropout=0.1, activation='relu'):
        """
        Args:
            in_dim: 输入维度
            out_dim: 输出维度
            dropout: Dropout比率
            activation: 激活函数类型
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # 主路径
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        
        self.dropout = nn.Dropout(dropout)
        
        # 残差连接（如果维度不同，需要投影）
        if in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)
        else:
            self.residual_proj = None
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.fc1.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        
        if self.residual_proj is not None:
            nn.init.xavier_uniform_(self.residual_proj.weight)
            if self.residual_proj.bias is not None:
                nn.init.zeros_(self.residual_proj.bias)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch, in_dim]
            
        Returns:
            output: 输出特征 [batch, out_dim]
        """
        # 主路径
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # 残差连接
        if self.residual_proj is not None:
            residual = self.residual_proj(x)
        else:
            residual = x
        
        output = out + residual
        
        return output


class SimpleTripletNetwork(nn.Module):
    """
    简单三元组网络：使用图像特征+姿势特征生成正面特征+正面姿势
    
    架构：
    1. 图像特征（src） + 姿势特征（pose） -> 连接 -> 融合特征
    2. 带残差的全连接层 -> 正面特征（front_features）
    3. 带残差的全连接层 -> 正面姿势（front_pose）
    
    用于三元组损失训练，代替复杂的Transformer矫正网络
    """
    def __init__(
        self,
        image_dim: int = 512,  # 图像特征维度（InsightFace: 512）
        pose_dim: int = 3,    # 姿势维度（欧拉角: 3）
        hidden_dim: int = 1024,  # 隐藏层维度
        num_layers: int = 3,  # 全连接层数量
        dropout: float = 0.1,
        activation: str = 'relu',
        front_pose: Optional[torch.Tensor] = None  # 目标正面姿势（可选，如果为None则学习）
    ):
        """
        初始化简单三元组网络
        
        Args:
            image_dim: 图像特征维度
            pose_dim: 姿势维度
            hidden_dim: 隐藏层维度
            num_layers: 全连接层数量
            dropout: Dropout比率
            activation: 激活函数类型
            front_pose: 目标正面姿势（如果为None，则从数据中学习）
        """
        super().__init__()
        
        self.image_dim = image_dim
        self.pose_dim = pose_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 输入维度：图像特征 + 姿势特征
        input_dim = image_dim + pose_dim
        
        # ========== 特征融合层 ==========
        # 将图像特征和姿势特征融合
        self.fusion_layers = nn.ModuleList()
        
        # 第一层：融合输入
        self.fusion_layers.append(
            ResidualFCBlock(input_dim, hidden_dim, dropout, activation)
        )
        
        # 中间层
        for _ in range(num_layers - 2):
            self.fusion_layers.append(
                ResidualFCBlock(hidden_dim, hidden_dim, dropout, activation)
            )
        
        # ========== 正面特征生成分支 ==========
        # 生成正面图像特征
        self.front_feature_layers = nn.ModuleList()
        
        # 第一层
        self.front_feature_layers.append(
            ResidualFCBlock(hidden_dim, hidden_dim, dropout, activation)
        )
        
        # 中间层
        for _ in range(num_layers - 2):
            self.front_feature_layers.append(
                ResidualFCBlock(hidden_dim, hidden_dim, dropout, activation)
            )
        
        # 输出层：生成正面特征
        self.front_feature_output = nn.Sequential(
            nn.Linear(hidden_dim, image_dim),
            nn.BatchNorm1d(image_dim)
        )
        
        # ========== 正面姿势生成分支 ==========
        # 生成正面姿势
        self.front_pose_layers = nn.ModuleList()
        
        # 第一层
        self.front_pose_layers.append(
            ResidualFCBlock(hidden_dim, hidden_dim // 2, dropout, activation)
        )
        
        # 中间层
        for _ in range(num_layers - 2):
            self.front_pose_layers.append(
                ResidualFCBlock(hidden_dim // 2, hidden_dim // 2, dropout, activation)
            )
        
        # 输出层：生成正面姿势
        self.front_pose_output = nn.Sequential(
            nn.Linear(hidden_dim // 2, pose_dim),
            nn.Tanh()  # 限制在[-1, 1]范围（欧拉角通常在这个范围）
        )
        
        # ========== 身份投影头（用于三元组损失） ==========
        # 将正面特征投影到身份特征空间
        self.identity_head = nn.Sequential(
            nn.Linear(image_dim, image_dim),
            nn.BatchNorm1d(image_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(image_dim, image_dim),
            nn.BatchNorm1d(image_dim)
        )
        
        # 初始化
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
        src: torch.Tensor,  # 图像特征 [batch, image_dim]
        pose: torch.Tensor,  # 姿势特征 [batch, pose_dim]
        return_identity_features: bool = True,  # 是否返回身份特征（用于三元组损失）
        return_front_pose: bool = True  # 是否返回正面姿势
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            src: 图像特征 [batch, image_dim]
            pose: 姿势特征 [batch, pose_dim]
            return_identity_features: 是否返回身份特征（用于三元组损失）
            return_front_pose: 是否返回正面姿势
            
        Returns:
            front_features: 正面图像特征 [batch, image_dim]
            identity_features: 归一化的身份特征 [batch, image_dim]（如果return_identity_features=True）
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
        
        # ========== 阶段2：生成正面特征 ==========
        # 通过正面特征生成层
        front_feat_hidden = fused_features
        for layer in self.front_feature_layers:
            front_feat_hidden = layer(front_feat_hidden)  # [batch, hidden_dim]
        
        # 生成正面特征
        front_features = self.front_feature_output(front_feat_hidden)  # [batch, image_dim]
        
        # 残差连接：正面特征 = 原始特征 + 生成的残差
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
            identity_features = self.identity_head(front_features)  # [batch, image_dim]
            
            # L2归一化（用于余弦相似度计算）
            identity_features = F.normalize(identity_features, p=2, dim=1)
        
        return front_features, identity_features, front_pose
    
    def get_trainable_parameters(self):
        """获取可训练参数"""
        return list(self.parameters())


class SimpleTripletNetworkWithProjection(nn.Module):
    """
    带投影头的简单三元组网络
    在正面特征基础上，进一步投影到身份特征空间
    """
    def __init__(
        self,
        image_dim: int = 512,
        pose_dim: int = 3,
        hidden_dim: int = 1024,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = 'relu',
        identity_dim: int = 512  # 身份特征维度
    ):
        """
        初始化带投影头的简单三元组网络
        
        Args:
            image_dim: 图像特征维度
            pose_dim: 姿势维度
            hidden_dim: 隐藏层维度
            num_layers: 全连接层数量
            dropout: Dropout比率
            activation: 激活函数类型
            identity_dim: 身份特征维度
        """
        super().__init__()
        
        # 基础网络
        self.base_network = SimpleTripletNetwork(
            image_dim=image_dim,
            pose_dim=pose_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation
        )
        
        # 额外的身份投影头
        self.identity_projection = nn.Sequential(
            nn.Linear(image_dim, identity_dim),
            nn.BatchNorm1d(identity_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(identity_dim, identity_dim),
            nn.BatchNorm1d(identity_dim)
        )
        
        self.identity_dim = identity_dim
    
    def forward(
        self,
        src: torch.Tensor,
        pose: torch.Tensor,
        return_identity_features: bool = True,
        return_front_pose: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            src: 图像特征 [batch, image_dim]
            pose: 姿势特征 [batch, pose_dim]
            return_identity_features: 是否返回身份特征
            return_front_pose: 是否返回正面姿势
            
        Returns:
            front_features: 正面图像特征 [batch, image_dim]
            identity_features: 归一化的身份特征 [batch, identity_dim]
            front_pose: 正面姿势 [batch, pose_dim]
        """
        # 基础网络前向传播
        front_features, _, front_pose = self.base_network(
            src=src,
            pose=pose,
            return_identity_features=False,  # 基础网络不返回身份特征
            return_front_pose=return_front_pose
        )
        
        # 生成身份特征
        identity_features = None
        if return_identity_features:
            identity_features = self.identity_projection(front_features)
            identity_features = F.normalize(identity_features, p=2, dim=1)
        
        return front_features, identity_features, front_pose


if __name__ == "__main__":
    """测试代码"""
    print("=" * 70)
    print("测试简单三元组网络")
    print("=" * 70)
    
    # 创建模型
    model = SimpleTripletNetwork(
        image_dim=512,
        pose_dim=3,
        hidden_dim=1024,
        num_layers=3,
        dropout=0.1
    )
    
    # 测试输入
    batch_size = 8
    src = torch.randn(batch_size, 512)
    pose = torch.randn(batch_size, 3)
    
    # 前向传播
    front_features, identity_features, front_pose = model(
        src=src,
        pose=pose,
        return_identity_features=True,
        return_front_pose=True
    )
    
    print(f"输入图像特征形状: {src.shape}")
    print(f"输入姿势特征形状: {pose.shape}")
    print(f"输出正面特征形状: {front_features.shape}")
    print(f"输出身份特征形状: {identity_features.shape}")
    print(f"输出正面姿势形状: {front_pose.shape}")
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数统计:")
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    print("\n✓ 测试通过！")

