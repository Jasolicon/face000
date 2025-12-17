"""
ControlNet-like 工具函数
"""
import torch
import torch.nn as nn
import numpy as np


class ZeroConv1d(nn.Module):
    """
    零卷积层（1D）
    用于 ControlNet-like 架构，确保训练初期不影响主网络
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)
        # 初始化为零
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, x):
        return self.conv(x)


class ZeroLinear(nn.Module):
    """
    零线性层
    用于 ControlNet-like 架构，确保训练初期不影响主网络
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # 初始化为零
        nn.init.zeros_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        return self.linear(x)


def compute_identity_consistency_loss(features_list, temperature=0.1):
    """
    计算身份一致性损失
    确保不同角度的特征属于同一身份
    
    Args:
        features_list: 特征列表，每个元素是 [batch, feature_dim]
        temperature: 温度参数，用于对比学习
    
    Returns:
        loss: 身份一致性损失
    """
    if len(features_list) < 2:
        return torch.tensor(0.0, device=features_list[0].device)
    
    # 归一化特征
    normalized_features = [torch.nn.functional.normalize(f, p=2, dim=1) for f in features_list]
    
    # 计算所有特征对之间的相似度
    total_loss = 0.0
    count = 0
    
    for i in range(len(normalized_features)):
        for j in range(i + 1, len(normalized_features)):
            # 余弦相似度
            similarity = (normalized_features[i] * normalized_features[j]).sum(dim=1).mean()
            # 损失：1 - 相似度（我们希望相似度接近1）
            loss = 1 - similarity
            total_loss += loss
            count += 1
    
    return total_loss / count if count > 0 else torch.tensor(0.0, device=features_list[0].device)

