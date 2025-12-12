"""
损失函数定义
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from train_transformer.utils_seed import set_seed

# 设置随机种子
set_seed(42)


class CosineSimilarityLoss(nn.Module):
    """
    余弦相似度损失
    loss = 1 - cosine_similarity(pred, target)
    """
    
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算余弦相似度损失
        
        Args:
            pred: 预测特征 [batch_size, feature_dim]
            target: 目标特征 [batch_size, feature_dim]
            
        Returns:
            loss: 损失值
        """
        # 归一化特征
        pred_norm = F.normalize(pred, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)
        
        # 计算余弦相似度
        cosine_sim = (pred_norm * target_norm).sum(dim=1).mean()
        
        # 损失 = 1 - 相似度
        loss = 1 - cosine_sim
        
        return loss


class MSELoss(nn.Module):
    """
    均方误差损失
    """
    
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算MSE损失
        
        Args:
            pred: 预测特征 [batch_size, feature_dim]
            target: 目标特征 [batch_size, feature_dim]
            
        Returns:
            loss: 损失值
        """
        return self.mse_loss(pred, target)


class CombinedLoss(nn.Module):
    """
    组合损失函数
    余弦相似度损失 + MSE损失
    """
    
    def __init__(self, cosine_weight: float = 0.5, mse_weight: float = 0.5):
        """
        初始化组合损失
        
        Args:
            cosine_weight: 余弦相似度损失的权重
            mse_weight: MSE损失的权重
        """
        super(CombinedLoss, self).__init__()
        self.cosine_loss = CosineSimilarityLoss()
        self.mse_loss = MSELoss()
        self.cosine_weight = cosine_weight
        self.mse_weight = mse_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        计算组合损失
        
        Args:
            pred: 预测特征 [batch_size, feature_dim]
            target: 目标特征 [batch_size, feature_dim]
            
        Returns:
            loss_dict: 包含总损失和各项损失的字典
        """
        cosine_loss = self.cosine_loss(pred, target)
        mse_loss = self.mse_loss(pred, target)
        
        total_loss = self.cosine_weight * cosine_loss + self.mse_weight * mse_loss
        
        return {
            'total_loss': total_loss,
            'cosine_loss': cosine_loss,
            'mse_loss': mse_loss
        }


class CosineSimilarityWithMarginLoss(nn.Module):
    """
    带边距的余弦相似度损失
    鼓励预测特征与目标特征的相似度大于某个阈值
    """
    
    def __init__(self, margin: float = 0.9):
        """
        初始化带边距的余弦相似度损失
        
        Args:
            margin: 相似度阈值（默认0.9）
        """
        super(CosineSimilarityWithMarginLoss, self).__init__()
        self.margin = margin
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算带边距的余弦相似度损失
        
        Args:
            pred: 预测特征 [batch_size, feature_dim]
            target: 目标特征 [batch_size, feature_dim]
            
        Returns:
            loss: 损失值
        """
        # 归一化特征
        pred_norm = F.normalize(pred, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)
        
        # 计算余弦相似度
        cosine_sim = (pred_norm * target_norm).sum(dim=1)
        
        # 损失 = max(0, margin - cosine_sim) 的平均值
        loss = F.relu(self.margin - cosine_sim).mean()
        
        return loss


class ResidualAndFinalLoss(nn.Module):
    """
    残差和最终特征组合损失
    同时优化残差预测和最终特征相似度
    这是改进版损失函数，解决损失下降但相似度提升缓慢的问题
    """
    
    def __init__(
        self, 
        residual_weight: float = 0.5,
        final_weight: float = 0.5,
        cosine_weight: float = 0.5,
        mse_weight: float = 0.5
    ):
        """
        初始化残差和最终特征组合损失
        
        Args:
            residual_weight: 残差损失的权重（默认0.5）
            final_weight: 最终特征损失的权重（默认0.5）
            cosine_weight: 余弦损失的权重（在残差和最终特征损失中都使用）
            mse_weight: MSE损失的权重（在残差和最终特征损失中都使用）
        """
        super(ResidualAndFinalLoss, self).__init__()
        self.residual_weight = residual_weight
        self.final_weight = final_weight
        self.cosine_loss = CosineSimilarityLoss()
        self.mse_loss = MSELoss()
        self.cosine_weight = cosine_weight
        self.mse_weight = mse_weight
    
    def forward(
        self, 
        predicted_residual: torch.Tensor,
        target_residual: torch.Tensor,
        corrected_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> dict:
        """
        计算组合损失
        
        Args:
            predicted_residual: 预测的残差 [batch_size, feature_dim]
            target_residual: 真实的残差 [batch_size, feature_dim]
            corrected_features: 矫正后的特征 [batch_size, feature_dim] (input + predicted_residual)
            target_features: 目标特征 [batch_size, feature_dim]
            
        Returns:
            loss_dict: 包含总损失和各项损失的字典
        """
        # 1. 残差损失（预测残差 vs 真实残差）
        residual_cosine_loss = self.cosine_loss(predicted_residual, target_residual)
        residual_mse_loss = self.mse_loss(predicted_residual, target_residual)
        residual_loss = self.cosine_weight * residual_cosine_loss + self.mse_weight * residual_mse_loss
        
        # 2. 最终特征损失（矫正后的特征 vs 目标特征）
        final_cosine_loss = self.cosine_loss(corrected_features, target_features)
        final_mse_loss = self.mse_loss(corrected_features, target_features)
        final_loss = self.cosine_weight * final_cosine_loss + self.mse_weight * final_mse_loss
        
        # 3. 组合损失
        total_loss = self.residual_weight * residual_loss + self.final_weight * final_loss
        
        return {
            'total_loss': total_loss,
            'residual_loss': residual_loss,
            'residual_cosine_loss': residual_cosine_loss,
            'residual_mse_loss': residual_mse_loss,
            'final_loss': final_loss,
            'final_cosine_loss': final_cosine_loss,
            'final_mse_loss': final_mse_loss
        }

