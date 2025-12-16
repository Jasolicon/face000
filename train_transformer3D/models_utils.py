"""
3D Transformer模型的工具类
包含角度位置编码、角度条件归一化等工具类
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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

