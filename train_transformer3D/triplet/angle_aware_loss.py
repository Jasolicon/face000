"""
角度感知三元组损失（Angle-Aware Triplet Loss）
用于人脸角度转换任务，强化模型对身份的聚焦能力

核心思想：强制模型拉近同一身份不同角度的特征，推开同一角度不同身份的特征。

参考论文：Domain-aware triplet loss in domain generalization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))


class AngleAwareTripletSampler:
    """
    角度感知三元组采样器
    根据角度信息，智能地构建困难且有教学意义的训练三元组。
    """
    
    def __init__(self, angle_threshold: float = 30.0):
        """
        Args:
            angle_threshold: 角度差异阈值（度），用于判断两个样本是否属于"相似角度"。
        """
        self.angle_threshold = angle_threshold
        
    def get_angle_difference(self, angle1: torch.Tensor, angle2: torch.Tensor) -> torch.Tensor:
        """计算两个角度向量之间的差异（欧氏距离）"""
        return torch.norm(angle1 - angle2, dim=1)
    
    def sample(self, features: torch.Tensor, labels: torch.Tensor, angles: torch.Tensor):
        """
        为一批数据采样角度感知三元组
        
        Args:
            features: 特征向量 [batch_size, feature_dim]
            labels: 身份标签 [batch_size]
            angles: 角度向量 [batch_size, angle_dim]
            
        Returns:
            triplets: 三元组索引 [num_triplets, 3] (anchor, positive, negative)
        """
        batch_size = features.size(0)
        device = features.device
        
        # 1. 计算角度距离矩阵
        angle_dist_matrix = torch.cdist(angles, angles, p=2)  # [batch_size, batch_size]
        
        # 2. 构建三元组列表
        triplets = []
        
        for i in range(batch_size):
            current_label = labels[i]
            current_angle = angles[i]
            
            # 寻找正样本：同身份、不同角度（跨角度）
            # 这里寻找"角度差异足够大"的同身份样本
            same_label_mask = (labels == current_label)
            same_label_mask[i] = False  # 排除自己
            
            if same_label_mask.any():
                same_label_indices = torch.where(same_label_mask)[0]
                # 计算与当前样本的角度差异
                angle_diffs = angle_dist_matrix[i, same_label_indices]
                # 选择角度差异最大的作为困难正样本
                hardest_pos_idx = same_label_indices[torch.argmax(angle_diffs)]
                hardest_pos_angle_diff = angle_diffs.max()
            else:
                # 如果没有同身份样本，跳过
                continue
            
            # 寻找负样本：不同身份、相似角度
            # 这里寻找"角度相似但身份不同"的困难负样本
            diff_label_mask = (labels != current_label)
            
            if diff_label_mask.any():
                diff_label_indices = torch.where(diff_label_mask)[0]
                # 计算角度差异
                neg_angle_diffs = angle_dist_matrix[i, diff_label_indices]
                # 找到角度差异小于阈值的（相似角度）
                similar_angle_mask = (neg_angle_diffs < self.angle_threshold)
                
                if similar_angle_mask.any():
                    similar_angle_indices = diff_label_indices[similar_angle_mask]
                    # 选择特征最相似的作为困难负样本
                    if similar_angle_indices.numel() > 0:
                        # 计算特征相似度
                        feature_similarities = F.cosine_similarity(
                            features[i].unsqueeze(0),
                            features[similar_angle_indices]
                        )
                        hardest_neg_idx = similar_angle_indices[torch.argmax(feature_similarities)]
                    else:
                        # 如果没有相似角度的，随机选一个不同身份的
                        hardest_neg_idx = diff_label_indices[torch.randint(0, diff_label_indices.size(0), (1,), device=device)]
                else:
                    # 没有相似角度的，选择角度最相似的
                    hardest_neg_idx = diff_label_indices[torch.argmin(neg_angle_diffs)]
            else:
                # 如果没有不同身份的样本，跳过
                continue
            
            triplets.append([i, hardest_pos_idx.item(), hardest_neg_idx.item()])
        
        if len(triplets) == 0:
            return torch.tensor([], dtype=torch.long, device=device)
        
        return torch.tensor(triplets, dtype=torch.long, device=device)


class AngleAwareTripletLoss(nn.Module):
    """
    角度感知三元组损失
    为不同类型的三元组分配不同的权重，强调跨角度身份聚合和同角度身份分离。
    """
    
    def __init__(self, margin: float = 0.2, alpha: float = 2.0, beta: float = 1.5, angle_threshold: float = 30.0):
        """
        Args:
            margin: 三元组损失的基础边界值
            alpha: 跨角度正样本对的权重因子（拉近不同角度的同一身份）
            beta: 同角度负样本对的权重因子（推开相同角度的不同身份）
            angle_threshold: 角度差异阈值（度）
        """
        super(AngleAwareTripletLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha  # 跨角度权重
        self.beta = beta    # 同角度权重
        self.sampler = AngleAwareTripletSampler(angle_threshold=angle_threshold)
        
    def compute_triplet_weight(self, anchor_angle: torch.Tensor, 
                               positive_angle: torch.Tensor,
                               negative_angle: torch.Tensor) -> float:
        """
        根据角度关系计算三元组权重
        返回权重值，范围通常在[1.0, alpha或beta]
        """
        # 计算角度差异
        pos_angle_diff = torch.norm(anchor_angle - positive_angle, p=2).item()
        neg_angle_diff = torch.norm(anchor_angle - negative_angle, p=2).item()
        
        base_weight = 1.0
        
        # 规则1：如果正样本是跨角度的（角度差异大），增加权重
        if pos_angle_diff > 45.0:  # 假设45度为"大角度差异"阈值
            base_weight *= self.alpha
        
        # 规则2：如果负样本是同角度的（角度差异小），增加权重
        if neg_angle_diff < 15.0:  # 假设15度为"小角度差异"阈值
            base_weight *= self.beta
        
        return base_weight
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor, 
                angles: torch.Tensor, features_orig: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        计算角度感知三元组损失
        
        Args:
            features: 模型输出的特征 [batch_size, feature_dim]
            labels: 身份标签 [batch_size]
            angles: 角度向量 [batch_size, angle_dim]
            features_orig: 原始侧面特征（可选，用于计算重建损失）
            
        Returns:
            total_loss: 总损失值
            loss_dict: 各损失分量的字典
        """
        batch_size = features.size(0)
        
        # 1. 采样三元组
        triplets = self.sampler.sample(features, labels, angles)
        
        if triplets.numel() == 0:
            # 如果没有有效的三元组，返回零损失
            return torch.tensor(0.0, device=features.device, requires_grad=True), {
                'triplet_loss': 0.0,
                'reconstruction_loss': 0.0,
                'total_loss': 0.0,
                'num_triplets': 0,
                'avg_weight': 0.0,
                'avg_pos_dist': 0.0,
                'avg_neg_dist': 0.0
            }
        
        # 2. 提取三元组特征
        anchors = features[triplets[:, 0]]
        positives = features[triplets[:, 1]]
        negatives = features[triplets[:, 2]]
        
        # 3. 提取对应的角度
        anchor_angles = angles[triplets[:, 0]]
        positive_angles = angles[triplets[:, 1]]
        negative_angles = angles[triplets[:, 2]]
        
        # 4. 计算基础三元组距离
        pos_dist = F.pairwise_distance(anchors, positives, p=2)
        neg_dist = F.pairwise_distance(anchors, negatives, p=2)
        
        # 5. 为每个三元组计算权重
        weights = []
        for i in range(len(triplets)):
            weight = self.compute_triplet_weight(
                anchor_angles[i],
                positive_angles[i],
                negative_angles[i]
            )
            weights.append(weight)
        
        weights = torch.tensor(weights, device=features.device, dtype=features.dtype)
        
        # 6. 计算加权损失
        basic_losses = F.relu(pos_dist - neg_dist + self.margin)
        weighted_losses = weights * basic_losses
        
        # 7. 计算总损失（平均）
        triplet_loss = weighted_losses.mean()
        
        # 8. 可选：添加重建损失（如果提供了原始特征）
        reconstruction_loss = torch.tensor(0.0, device=features.device)
        if features_orig is not None:
            # 假设我们希望正面化特征与原始侧面特征在身份上相关但不相同
            # 使用余弦相似度作为重建约束
            cos_sim = F.cosine_similarity(features, features_orig, dim=1).mean()
            reconstruction_loss = 1.0 - cos_sim  # 我们希望最大化相似度，所以最小化1-sim
        
        # 9. 组合总损失
        total_loss = triplet_loss + 0.1 * reconstruction_loss
        
        # 10. 返回损失详情
        loss_dict = {
            'triplet_loss': triplet_loss.item(),
            'reconstruction_loss': reconstruction_loss.item() if features_orig is not None else 0.0,
            'total_loss': total_loss.item(),
            'num_triplets': len(triplets),
            'avg_weight': weights.mean().item(),
            'avg_pos_dist': pos_dist.mean().item(),
            'avg_neg_dist': neg_dist.mean().item()
        }
        
        return total_loss, loss_dict


if __name__ == "__main__":
    """测试代码"""
    print("=" * 70)
    print("角度感知三元组损失测试")
    print("=" * 70)
    
    # 创建损失函数
    loss_fn = AngleAwareTripletLoss(margin=0.2, alpha=2.0, beta=1.5)
    
    # 创建模拟数据
    batch_size = 16
    feature_dim = 768
    angle_dim = 5
    
    features = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, 4, (batch_size,))  # 4个不同身份
    angles = torch.randn(batch_size, angle_dim) * 30.0  # 角度范围约-30到30度
    features_orig = torch.randn(batch_size, feature_dim)
    
    print(f"\n输入数据:")
    print(f"  特征形状: {features.shape}")
    print(f"  标签形状: {labels.shape}")
    print(f"  角度形状: {angles.shape}")
    print(f"  身份数量: {len(torch.unique(labels))}")
    
    # 计算损失
    print("\n计算损失...")
    total_loss, loss_dict = loss_fn(features, labels, angles, features_orig)
    
    print(f"\n损失结果:")
    for key, value in loss_dict.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)

