"""
先验保护损失：防止微调时丢失原始生成能力
参考ViewDiff和DreamBooth的先验保护策略
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class PriorPreservationLoss(nn.Module):
    """
    先验保护损失：防止微调时丢失原始生成能力
    
    核心思想：
    - 保持当前模型输出与基础模型输出的相似性
    - 通过KL散度、特征相似度、L2距离等多重约束
    - 参考DreamBooth和ViewDiff的先验保护策略
    """
    def __init__(self, base_model, lambda_prior=0.1):
        """
        Args:
            base_model: 原始预训练模型（冻结）
            lambda_prior: 先验损失权重
        """
        super().__init__()
        self.base_model = base_model
        self.lambda_prior = lambda_prior
        
        # 冻结基础模型
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        self.base_model.eval()
        
        # 创建先验保护数据集（缓存）
        self.prior_cache = {}
        
    def cache_prior_samples(self, dataloader, num_samples=100, device='cuda'):
        """
        缓存原始模型的生成样本作为先验参考
        
        Args:
            dataloader: 数据加载器
            num_samples: 缓存样本数量
            device: 设备
        """
        self.base_model.eval()
        prior_samples = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if len(prior_samples) >= num_samples:
                    break
                
                # 假设batch包含输入和条件
                # 根据实际数据格式调整
                if isinstance(batch, dict):
                    src = batch['src'].to(device)
                    pose = batch.get('pose', None)
                    if pose is not None:
                        pose = pose.to(device)
                    angles = batch.get('angles', pose)
                    if angles is not None:
                        angles = angles.to(device)
                else:
                    src, angles = batch[0].to(device), batch[1].to(device)
                    pose = None
                
                # 使用基础模型生成
                try:
                    outputs = self.base_model(src, angles=angles, pose=pose)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    prior_samples.append(outputs.detach().cpu())
                except Exception as e:
                    print(f"警告: 缓存先验样本时出错: {e}")
                    continue
        
        if prior_samples:
            self.prior_cache['samples'] = torch.cat(prior_samples, dim=0)[:num_samples]
            print(f"✓ 已缓存 {len(self.prior_cache['samples'])} 个先验样本")
        
    def compute_kl_divergence(self, current_output, prior_output):
        """
        计算KL散度作为先验损失
        
        Args:
            current_output: 当前模型输出 [batch, ...]
            prior_output: 基础模型输出 [batch, ...]
            
        Returns:
            kl_loss: KL散度损失
        """
        # 展平特征
        current_flat = current_output.flatten(1)  # [batch, features]
        prior_flat = prior_output.flatten(1)
        
        # 计算均值和标准差
        current_mean = current_flat.mean(dim=0)
        current_std = current_flat.std(dim=0) + 1e-8
        
        prior_mean = prior_flat.mean(dim=0)
        prior_std = prior_flat.std(dim=0) + 1e-8
        
        # 计算两个高斯分布之间的KL散度
        # KL(N(μ1, σ1) || N(μ2, σ2)) = log(σ2/σ1) + (σ1^2 + (μ1-μ2)^2) / (2*σ2^2) - 0.5
        kl = torch.log(prior_std / current_std) + \
             (current_std**2 + (current_mean - prior_mean)**2) / (2 * prior_std**2) - 0.5
        
        return kl.mean()
    
    def compute_feature_similarity(self, current_features, prior_features):
        """
        计算特征相似性损失
        
        Args:
            current_features: 当前模型特征 [batch, ...]
            prior_features: 基础模型特征 [batch, ...]
            
        Returns:
            similarity_loss: 相似度损失（越小越相似）
        """
        # 展平特征
        current_flat = current_features.flatten(1)
        prior_flat = prior_features.flatten(1)
        
        # 余弦相似度
        current_norm = F.normalize(current_flat, p=2, dim=-1)
        prior_norm = F.normalize(prior_flat, p=2, dim=-1)
        
        cosine_sim = (current_norm * prior_norm).sum(dim=-1).mean()
        return 1 - cosine_sim  # 最小化时使相似度接近1
    
    def forward(self, current_model, inputs, conditions, original_loss):
        """
        计算带先验保护的总损失
        
        Args:
            current_model: 当前正在微调的模型
            inputs: 输入数据
            conditions: 条件（如姿态）
            original_loss: 原始任务损失
            
        Returns:
            total_loss: 总损失
            loss_dict: 损失详情字典
        """
        # 1. 计算当前模型输出
        current_output = current_model(inputs, **conditions)
        if isinstance(current_output, tuple):
            current_output = current_output[0]
        
        # 2. 计算基础模型输出（先验）
        with torch.no_grad():
            prior_output = self.base_model(inputs, **conditions)
            if isinstance(prior_output, tuple):
                prior_output = prior_output[0]
        
        # 3. 计算先验保护损失
        # 选项1：KL散度
        kl_loss = self.compute_kl_divergence(current_output, prior_output)
        
        # 选项2：特征相似度
        feature_loss = self.compute_feature_similarity(current_output, prior_output)
        
        # 选项3：L2距离
        l2_loss = F.mse_loss(current_output, prior_output)
        
        # 组合损失
        prior_loss = kl_loss + feature_loss + l2_loss
        
        # 4. 总损失 = 原始损失 + λ * 先验损失
        total_loss = original_loss + self.lambda_prior * prior_loss
        
        return total_loss, {
            'original_loss': original_loss.item() if torch.is_tensor(original_loss) else original_loss,
            'prior_loss': prior_loss.item(),
            'kl_loss': kl_loss.item(),
            'feature_loss': feature_loss.item(),
            'l2_loss': l2_loss.item(),
            'total_loss': total_loss.item()
        }


class PriorPreservationDataset:
    """
    先验保护数据集管理器
    
    用于生成和缓存先验样本，在训练时与真实数据混合
    """
    def __init__(self, base_model, cache_size=100):
        """
        Args:
            base_model: 基础模型（冻结）
            cache_size: 缓存大小
        """
        self.base_model = base_model
        self.cache_size = cache_size
        self.cache = []
        
        # 冻结基础模型
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()
        
    def generate_prior_batch(self, conditions, batch_size=8, device='cuda'):
        """
        生成先验批次
        
        Args:
            conditions: 条件字典
            batch_size: 批次大小
            device: 设备
            
        Returns:
            prior_output: 先验输出 [batch_size, ...]
        """
        if len(self.cache) >= self.cache_size:
            # 从缓存中采样
            indices = torch.randint(0, len(self.cache), (batch_size,))
            return torch.stack([self.cache[i] for i in indices]).to(device)
        
        # 生成新样本
        with torch.no_grad():
            # 生成伪输入（如随机特征）
            # 假设特征维度是512（根据实际情况调整）
            pseudo_input = torch.randn(batch_size, 512).to(device)
            
            # 从条件中提取姿态信息
            pose = conditions.get('pose', None)
            if pose is None:
                pose = conditions.get('angles', None)
            if pose is not None:
                pose = pose[:batch_size].to(device)
            
            # 使用基础模型生成
            try:
                prior_output = self.base_model(
                    pseudo_input,
                    angles=pose if pose is not None else None,
                    pose=pose
                )
                if isinstance(prior_output, tuple):
                    prior_output = prior_output[0]
                
                # 添加到缓存
                self.cache.extend([p.cpu() for p in prior_output])
                if len(self.cache) > self.cache_size:
                    self.cache = self.cache[-self.cache_size:]
                
                return prior_output
            except Exception as e:
                print(f"警告: 生成先验批次时出错: {e}")
                # 返回零张量作为fallback
                return torch.zeros(batch_size, 512).to(device)

