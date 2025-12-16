"""
损失函数模块
提供各种损失函数用于训练
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    """
    均方误差损失
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测值 [batch, ...]
            target: 目标值 [batch, ...]
        Returns:
            loss: 均方误差损失
        """
        return self.criterion(pred, target)


class CosineSimilarityLoss(nn.Module):
    """
    余弦相似度损失（1 - 余弦相似度）
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测值 [batch, feature_dim]
            target: 目标值 [batch, feature_dim]
        Returns:
            loss: 余弦相似度损失（1 - 余弦相似度）
        """
        # 归一化
        pred_norm = F.normalize(pred, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)
        
        # 计算余弦相似度
        cosine_sim = (pred_norm * target_norm).sum(dim=1).mean()
        
        # 返回损失（1 - 余弦相似度）
        return 1 - cosine_sim


class CombinedLoss(nn.Module):
    """
    组合损失：MSE + 余弦相似度损失
    """
    def __init__(self, mse_weight=0.5, cosine_weight=0.5):
        """
        Args:
            mse_weight: MSE损失权重
            cosine_weight: 余弦相似度损失权重
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.mse_loss = MSELoss()
        self.cosine_loss = CosineSimilarityLoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测值 [batch, feature_dim]
            target: 目标值 [batch, feature_dim]
        Returns:
            loss: 组合损失
        """
        mse = self.mse_loss(pred, target)
        cosine = self.cosine_loss(pred, target)
        
        return self.mse_weight * mse + self.cosine_weight * cosine

