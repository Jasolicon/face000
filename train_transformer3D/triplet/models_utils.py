"""
三元组损失模块的工具类
包含角度位置编码、角度条件归一化和姿态条件注意力等工具类
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np


class AnglePositionalEncoding(nn.Module):
    """
    基于球面角的位置编码
    将角度转换为位置编码
    """
    
    def __init__(self, d_model: int, angle_dim: int = 5):
        """
        初始化角度位置编码
        
        Args:
            d_model: 模型维度
            angle_dim: 角度维度（关键点数量，默认5）
        """
        super(AnglePositionalEncoding, self).__init__()
        
        self.angle_dim = angle_dim
        self.d_model = d_model
        
        # 将角度映射到d_model维度的线性层
        self.angle_projection = nn.Linear(angle_dim, d_model)
        
        # 可学习的缩放因子
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """
        将角度转换为位置编码
        
        Args:
            angles: 角度张量 [batch_size, angle_dim] 或 [batch_size, seq_len, angle_dim]
            
        Returns:
            pos_encoding: 位置编码 [batch_size, d_model] 或 [batch_size, seq_len, d_model]
        """
        # 投影到d_model维度
        pos_encoding = self.angle_projection(angles)  # [B, angle_dim] -> [B, d_model]
        
        # 应用缩放
        pos_encoding = pos_encoding * self.scale
        
        return pos_encoding


class AngleConditionedLayerNorm(nn.Module):
    """
    角度条件归一化层
    使用角度信息来调制特征的归一化，让角度更深入地影响特征变换
    """
    def __init__(self, d_model: int, angle_dim: int = 5):
        super(AngleConditionedLayerNorm, self).__init__()
        self.d_model = d_model
        self.angle_dim = angle_dim
        
        # 角度到缩放和偏移的映射
        self.angle_to_scale = nn.Sequential(
            nn.Linear(angle_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()  # 缩放因子在0-1之间
        )
        self.angle_to_shift = nn.Sequential(
            nn.Linear(angle_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # 标准LayerNorm
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [batch_size, d_model] 或 [batch_size, seq_len, d_model]
            angles: 角度 [batch_size, angle_dim] 或 [batch_size, seq_len, angle_dim]
        Returns:
            条件归一化后的特征
        """
        # 计算角度相关的缩放和偏移
        scale = self.angle_to_scale(angles)  # [batch_size, d_model] 或 [batch_size, seq_len, d_model]
        shift = self.angle_to_shift(angles)  # [batch_size, d_model] 或 [batch_size, seq_len, d_model]
        
        # 标准归一化
        x_norm = self.layer_norm(x)
        
        # 应用角度相关的缩放和偏移
        x_conditioned = x_norm * (1 + scale) + shift
        
        return x_conditioned


class PoseConditionedAttention(nn.Module):
    """
    姿态条件注意力：将头部姿态（旋转向量/欧拉角）作为键值对，
    让特征（查询）去关注相关的姿态信息
    """
    def __init__(self, feature_dim, pose_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.pose_dim = pose_dim
        
        # 将姿态编码为键和值
        self.pose_to_key = nn.Linear(pose_dim, feature_dim)
        self.pose_to_value = nn.Linear(pose_dim, feature_dim)
        
        # 将特征作为查询
        self.feature_to_query = nn.Linear(feature_dim, feature_dim)
        
        # 注意力缩放因子
        self.scale = feature_dim ** -0.5
        
    def forward(self, features, pose):
        """
        features: [batch, seq_len, feature_dim] 或 [batch, feature_dim]
        pose: [batch, pose_dim]  # 可以是欧拉角、旋转向量等
        """
        if len(features.shape) == 2:
            features = features.unsqueeze(1)  # [batch, 1, feature_dim]
        
        batch_size, seq_len, _ = features.shape
        
        # 生成查询、键、值
        queries = self.feature_to_query(features)  # [batch, seq_len, feature_dim]
        
        # 姿态作为全局的键和值（同一姿态用于所有序列位置）
        keys = self.pose_to_key(pose).unsqueeze(1)  # [batch, 1, feature_dim]
        values = self.pose_to_value(pose).unsqueeze(1)  # [batch, 1, feature_dim]
        
        # 计算注意力分数
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 应用注意力
        attended = torch.matmul(attention_weights, values)
        
        # 残差连接
        output = features + attended
        
        if seq_len == 1:
            output = output.squeeze(1)
            
        return output, attention_weights


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_deterministic_mode():
    """设置确定性模式"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

