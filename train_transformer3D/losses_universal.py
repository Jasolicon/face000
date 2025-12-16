"""
通用人脸网络的损失函数
融合多任务学习、对比学习和特征解耦约束
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class UniversalFaceLoss(nn.Module):
    """多任务损失函数，综合各种约束"""
    
    def __init__(self, 
                 lambda_id=1.0,
                 lambda_pose=0.5,
                 lambda_ortho=0.1,
                 lambda_contrast=0.3,
                 lambda_reconstruction=0.2,
                 lambda_similarity_protection=0.5,
                 temperature=0.1,  # 从0.07增加到0.1，使对比学习更平滑
                 id_dim=256,
                 feat_dim=512):
        super().__init__()
        self.lambda_id = lambda_id
        self.lambda_pose = lambda_pose
        self.lambda_ortho = lambda_ortho
        self.lambda_contrast = lambda_contrast
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_similarity_protection = lambda_similarity_protection
        self.temperature = temperature
        
        # 各种损失函数
        self.pose_loss = nn.L1Loss()
        self.reconstruction_loss = nn.MSELoss()
        self.cosine_sim_loss = nn.CosineEmbeddingLoss()
        
        # 投影层（用于特征空间转换）
        self.id_to_feat_proj = nn.Linear(id_dim, feat_dim, bias=False)
        self.feat_to_id_proj = nn.Linear(feat_dim, id_dim, bias=False)
        
    def info_nce_loss(self, features, labels):
        """
        InfoNCE对比损失
        同一人的不同姿态应该相似，不同人的应该不同
        """
        features = F.normalize(features, dim=1)
        similarity = torch.mm(features, features.t()) / self.temperature
        
        # 创建正样本掩码（同一人）
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask = mask - torch.eye(len(labels), device=features.device)  # 排除自己
        
        # 计算对比损失
        exp_sim = torch.exp(similarity)
        pos_sum = torch.sum(exp_sim * mask, dim=1)
        neg_sum = torch.sum(exp_sim, dim=1) - pos_sum - 1  # 减去自己和正样本
        
        # 避免log(0)
        pos_sum = pos_sum + 1e-8
        neg_sum = neg_sum + 1e-8
        
        loss = -torch.log(pos_sum / (pos_sum + neg_sum))
        # 只对有正样本的样本计算损失
        valid_mask = mask.sum(dim=1) > 0
        if valid_mask.sum() > 0:
            return loss[valid_mask].mean()
        else:
            return torch.tensor(0.0, device=features.device)
    
    def pose_consistency_loss(self, pose_features, pose_angles, pose_prototypes):
        """
        姿态一致性损失：将姿态特征与姿态原型对齐
        """
        pose_features = F.normalize(pose_features, dim=1)
        pose_prototypes = F.normalize(pose_prototypes, dim=1)
        
        # 计算与所有原型的相似度
        similarity = torch.mm(pose_features, pose_prototypes.t())  # [batch, num_prototypes]
        
        # 根据角度计算应该对应的原型索引（简化：根据yaw角度）
        yaw = pose_angles[:, 0]  # [batch]
        # 将yaw映射到[0, num_prototypes-1]
        yaw_normalized = (yaw + 90) / 180.0  # 假设yaw范围是[-90, 90]
        yaw_normalized = torch.clamp(yaw_normalized, 0, 1)
        target_indices = (yaw_normalized * (pose_prototypes.shape[0] - 1)).long()
        
        # 使用交叉熵鼓励特征靠近对应原型
        loss = F.cross_entropy(similarity / 0.1, target_indices)
        return loss
    
    def forward(self, outputs: Dict, targets: Dict, model=None):
        """
        Args:
            outputs: 模型输出字典，包含：
                - id_features: [batch, id_dim]
                - pose_features: [batch, pose_dim]
                - pose_angles: [batch, 3] (估计的姿态)
                - base_features: [batch, feat_dim] (可选)
            targets: 目标字典，包含：
                - tgt_features: [batch, feat_dim] (目标特征，通常是正面特征)
                - pose_labels: [batch, 3] (真实姿态角度)
                - person_names: [batch] (人员ID，用于对比学习)
            model: 模型实例（用于获取正交损失）
        Returns:
            losses: 损失字典
        """
        losses = {}
        device = outputs['id_features'].device
        
        # 1. 重建损失：生成的正面特征应该接近目标特征
        if 'base_features' in outputs and 'tgt_features' in targets:
            # 使用身份特征重建正面特征
            tgt_features = targets['tgt_features']
            id_features = outputs['id_features']
            
            # 将id_features投影回feat_dim空间（简化重建）
            reconstructed = self.id_to_feat_proj(id_features)
            losses['reconstruction'] = self.reconstruction_loss(reconstructed, tgt_features)
        else:
            losses['reconstruction'] = torch.tensor(0.0, device=device)
        
        # 1.5. 相似度保护损失（新增：确保模型输出相似度不低于原始相似度）
        if 'base_features' in outputs and 'tgt_features' in targets and 'id_features' in outputs:
            # 计算原始相似度（base_features vs tgt_features）
            base_norm = F.normalize(outputs['base_features'], dim=1)
            tgt_norm = F.normalize(targets['tgt_features'], dim=1)
            original_sim = F.cosine_similarity(base_norm, tgt_norm, dim=1)
            
            # 计算模型输出相似度（id_features投影到512维 vs tgt_features）
            id_features_512 = self.id_to_feat_proj(outputs['id_features'])
            id_features_512_norm = F.normalize(id_features_512, dim=1)
            model_sim = F.cosine_similarity(id_features_512_norm, tgt_norm, dim=1)
            
            # 保护损失：如果模型相似度低于原始相似度，给予惩罚
            # 改进：添加margin，允许小的下降，减少不必要的惩罚
            margin = 0.01  # 允许0.01的下降
            protection_loss = F.relu(original_sim - model_sim - margin).mean()
            losses['similarity_protection'] = protection_loss
        else:
            losses['similarity_protection'] = torch.tensor(0.0, device=device)
        
        # 2. 姿态估计损失
        if 'pose_angles' in outputs and 'pose_labels' in targets:
            losses['pose'] = self.pose_loss(
                outputs['pose_angles'], 
                targets['pose_labels']
            )
        else:
            losses['pose'] = torch.tensor(0.0, device=device)
        
        # 3. 对比学习损失（同一人不同姿态）
        if 'id_features' in outputs and 'person_names' in targets:
            person_names = targets['person_names']
            # 将person_names转换为数字标签
            unique_names = list(set(person_names))
            name_to_label = {name: idx for idx, name in enumerate(unique_names)}
            labels = torch.tensor([name_to_label[name] for name in person_names], 
                                 device=device, dtype=torch.long)
            
            losses['contrast'] = self.info_nce_loss(
                outputs['id_features'],
                labels
            )
        else:
            losses['contrast'] = torch.tensor(0.0, device=device)
        
        # 4. 正交约束损失
        if model is not None:
            losses['ortho'] = model.get_ortho_loss()
        else:
            losses['ortho'] = torch.tensor(0.0, device=device)
        
        # 5. 姿态一致性损失（使用原型）
        if 'pose_features' in outputs and 'pose_labels' in targets and hasattr(model, 'pose_prototypes'):
            losses['pose_consistency'] = self.pose_consistency_loss(
                outputs['pose_features'],
                targets['pose_labels'],
                model.pose_prototypes
            )
        else:
            losses['pose_consistency'] = torch.tensor(0.0, device=device)
        
        # 6. 余弦相似度损失（改进：在512维空间计算，避免信息丢失）
        if 'id_features' in outputs and 'tgt_features' in targets:
            tgt_features = targets['tgt_features']
            
            # 改进：在512维空间计算相似度，而不是投影到256维
            # 将id_features投影回512维空间
            id_features_512 = self.id_to_feat_proj(outputs['id_features'])  # [batch, 256] -> [batch, 512]
            
            # 归一化
            id_features_512_norm = F.normalize(id_features_512, dim=1)
            tgt_features_norm = F.normalize(tgt_features, dim=1)
            
            # 计算余弦相似度（在512维空间）
            cosine_sim = F.cosine_similarity(id_features_512_norm, tgt_features_norm, dim=1)
            losses['id_similarity'] = (1 - cosine_sim).mean()
        else:
            losses['id_similarity'] = torch.tensor(0.0, device=device)
        
        # 加权总损失（改进：调整权重，更重视身份相似度）
        total_loss = (
            self.lambda_id * losses.get('id_similarity', 0) +
            self.lambda_pose * losses.get('pose', 0) +
            self.lambda_ortho * losses.get('ortho', 0) +
            self.lambda_contrast * losses.get('contrast', 0) +
            self.lambda_reconstruction * losses.get('reconstruction', 0) +
            0.1 * losses.get('pose_consistency', 0) +  # 较小的权重
            self.lambda_similarity_protection * losses.get('similarity_protection', 0)  # 相似度保护损失（可配置权重）
        )
        
        losses['total'] = total_loss
        return losses


if __name__ == "__main__":
    # 测试损失函数
    print("=" * 70)
    print("测试 UniversalFaceLoss")
    print("=" * 70)
    
    loss_fn = UniversalFaceLoss(
        lambda_id=1.0,
        lambda_pose=0.5,
        lambda_ortho=0.1,
        lambda_contrast=0.3,
        lambda_reconstruction=0.2
    )
    
    batch_size = 8
    feat_dim = 512
    id_dim = 256
    pose_dim = 128
    
    # 模拟模型输出
    outputs = {
        'id_features': torch.randn(batch_size, id_dim),
        'pose_features': torch.randn(batch_size, pose_dim),
        'pose_angles': torch.randn(batch_size, 3),
        'base_features': torch.randn(batch_size, feat_dim)
    }
    
    # 模拟目标
    targets = {
        'tgt_features': torch.randn(batch_size, feat_dim),
        'pose_labels': torch.randn(batch_size, 3),
        'person_names': ['person_0', 'person_0', 'person_1', 'person_1', 
                         'person_2', 'person_2', 'person_3', 'person_3']
    }
    
    # 模拟模型（用于正交损失）
    class MockModel:
        def get_ortho_loss(self):
            return torch.tensor(0.01)
        pose_prototypes = torch.randn(36, pose_dim)
    
    model = MockModel()
    
    # 计算损失
    losses = loss_fn(outputs, targets, model)
    
    print("\n各项损失:")
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.6f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
