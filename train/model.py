"""
多角度人脸识别模型
基于DINO + 孪生网络
"""
import os

# 在导入任何可能使用 HuggingFace 的库之前设置镜像
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import timm
import torchvision.transforms as transforms


class DINOFeatureExtractor(nn.Module):
    """DINO特征提取器（用于孪生网络）"""
    
    def __init__(self, model_name: str = 'vit_base_patch16_224', freeze_backbone: bool = False):
        """
        初始化DINO特征提取器
        
        Args:
            model_name: DINO模型名称
            freeze_backbone: 是否冻结backbone
        """
        super().__init__()
        
        # 确保使用镜像（在调用 timm.create_model 前）
        if 'HF_ENDPOINT' not in os.environ:
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        # 加载DINO模型
        # timm 会通过 huggingface_hub 下载，使用 HF_ENDPOINT 环境变量
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # 移除分类头，只使用特征提取
        )
        
        # 获取特征维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.backbone(dummy_input)
            self.feature_dim = dummy_output.shape[1]
        
        # 是否冻结backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 图像预处理（ImageNet标准）
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, C, H, W] (已归一化)
            
        Returns:
            特征向量 [B, D]
        """
        # 提取特征
        features = self.backbone(x)
        
        # L2归一化
        features = F.normalize(features, p=2, dim=1)
        
        return features


class SiameseNetwork(nn.Module):
    """孪生网络：共享权重的DINO特征提取器 + 相似度计算"""
    
    def __init__(
        self,
        dino_model_name: str = 'vit_base_patch16_224',
        feature_dim: int = 768,
        freeze_backbone: bool = False
    ):
        """
        初始化孪生网络
        
        Args:
            dino_model_name: DINO模型名称
            feature_dim: 特征维度（DINO输出维度）
            freeze_backbone: 是否冻结backbone
        """
        super().__init__()
        
        # 共享的特征提取器（孪生网络的核心）
        self.feature_extractor = DINOFeatureExtractor(
            model_name=dino_model_name,
            freeze_backbone=freeze_backbone
        )
        
        # 获取DINO实际输出维度
        dino_output_dim = self.feature_extractor.feature_dim
        
        # 特征投影层（使用DINO实际输出维度）
        self.feature_proj = nn.Sequential(
            nn.Linear(dino_output_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        提取特征
        
        Args:
            images: 输入图像 [B, C, H, W]
            
        Returns:
            特征向量 [B, D]
        """
        # 使用共享的特征提取器
        features = self.feature_extractor(images)
        
        # 特征投影
        features = self.feature_proj(features)
        
        # L2归一化
        features = F.normalize(features, p=2, dim=1)
        
        return features
    
    def compute_similarity(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """
        计算两个特征向量的相似度（余弦相似度）
        
        Args:
            features1: 特征1 [B, D]
            features2: 特征2 [B, D]
            
        Returns:
            相似度 [B]
        """
        # 余弦相似度
        similarity = F.cosine_similarity(features1, features2, dim=1)
        
        return similarity
    
    def forward(
        self,
        images1: torch.Tensor,
        images2: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            images1: 输入图像1 [B, C, H, W]
            images2: 输入图像2 [B, C, H, W]（可选）
            
        Returns:
            features1: 特征1 [B, D]
            similarity: 相似度 [B]（如果提供了images2）
        """
        # 提取特征1
        features1 = self.extract_features(images1)
        
        # 如果提供了images2，计算相似度
        similarity = None
        if images2 is not None:
            features2 = self.extract_features(images2)
            similarity = self.compute_similarity(features1, features2)
        
        return features1, similarity


class MultiAngleFaceModel(nn.Module):
    """多角度人脸识别模型（基于DINO孪生网络）"""
    
    def __init__(
        self,
        dino_model_name: str = 'vit_base_patch16_224',
        feature_dim: int = 768,
        num_classes: Optional[int] = None,
        margin: float = 0.5,
        scale: float = 64.0,
        freeze_backbone: bool = False
    ):
        """
        初始化模型
        
        Args:
            dino_model_name: DINO模型名称
            feature_dim: 输出特征维度（DINO默认768）
            num_classes: 类别数（用于ArcFace，如果为None则不使用分类头）
            margin: ArcFace margin参数
            scale: ArcFace scale参数
            freeze_backbone: 是否冻结DINO backbone
        """
        super().__init__()
        
        # 孪生网络（共享权重的特征提取器）
        self.siamese_net = SiameseNetwork(
            dino_model_name=dino_model_name,
            feature_dim=feature_dim,
            freeze_backbone=freeze_backbone
        )
        
        # ArcFace分类头（如果提供类别数）
        self.num_classes = num_classes
        if num_classes is not None:
            self.arcface_head = ArcFaceHead(
                in_features=feature_dim,
                out_features=num_classes,
                margin=margin,
                scale=scale
            )
        else:
            self.arcface_head = None
    
    def forward(
        self,
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        images2: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            images: 输入图像 [B, C, H, W]
            labels: 标签 [B]（可选，用于训练）
            images2: 第二个输入图像 [B, C, H, W]（可选，用于计算相似度）
            
        Returns:
            features: 特征向量 [B, D]
            logits: 分类logits [B, num_classes]（如果有分类头）
            similarity: 相似度 [B]（如果提供了images2）
        """
        # 提取特征
        features, similarity = self.siamese_net(images, images2)
        
        # ArcFace分类（如果提供标签）
        logits = None
        if self.arcface_head is not None and labels is not None:
            logits = self.arcface_head(features, labels)
        
        return features, logits, similarity
    
    def extract_feature(self, image: torch.Tensor) -> torch.Tensor:
        """
        提取特征（推理模式，兼容DINOFeatureExtractor）
        
        Args:
            image: 输入图像 [C, H, W] 或 [1, C, H, W]（已归一化）
            
        Returns:
            特征向量 [D] (numpy array)
        """
        self.eval()
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            
            features, _ = self.siamese_net(image)
            # 转换为numpy数组（兼容DINOFeatureExtractor）
            return features[0].cpu().numpy()
    
    def extract_features_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        批量提取特征
        
        Args:
            images: 输入图像 [B, C, H, W]
            
        Returns:
            特征向量 [B, D]
        """
        self.eval()
        with torch.no_grad():
            features, _ = self.siamese_net(images)
            return features


class ArcFaceHead(nn.Module):
    """ArcFace分类头"""
    
    def __init__(self, in_features: int, out_features: int, margin: float = 0.5, scale: float = 64.0):
        """
        初始化ArcFace头
        
        Args:
            in_features: 输入特征维度
            out_features: 输出类别数
            margin: margin参数（弧度）
            scale: scale参数
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.margin = margin
        self.scale = scale
        
        # 权重矩阵
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 特征向量 [B, D]，已归一化
            labels: 标签 [B]
            
        Returns:
            logits [B, num_classes]
        """
        # 归一化权重
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # 计算余弦相似度
        cosine = F.linear(features, weight)  # [B, num_classes]
        
        # 限制在[-1, 1]范围内（数值稳定性）
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # 计算角度
        theta = torch.acos(cosine)
        
        # 为正确类别添加margin
        target_theta = theta[torch.arange(len(labels)), labels].unsqueeze(1)
        target_theta_margin = target_theta + self.margin
        
        # 创建新的cosine值
        cosine_margin = torch.cos(target_theta_margin)
        
        # 只对正确类别应用margin
        output = cosine * 1.0
        output[torch.arange(len(labels)), labels] = cosine_margin.squeeze(1)
        
        # 应用scale
        output = output * self.scale
        
        return output


class ContrastiveLoss(nn.Module):
    """对比损失（用于正脸-多角度对）"""
    
    def __init__(self, margin: float = 0.2, temperature: float = 0.07):
        """
        初始化对比损失
        
        Args:
            margin: 正样本对的最小距离
            temperature: 温度参数
        """
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(
        self,
        front_features: torch.Tensor,
        angle_features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算对比损失
        
        Args:
            front_features: 正脸特征 [B, D]
            angle_features: 多角度特征 [B, D]
            labels: 标签 [B]
            
        Returns:
            损失值
        """
        # 计算相似度矩阵
        similarity = F.cosine_similarity(
            front_features.unsqueeze(1),
            angle_features.unsqueeze(0),
            dim=2
        )  # [B, B]
        
        # 创建正样本mask（同一人）
        positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        
        # 使用InfoNCE损失（对比学习标准损失）
        # 对于每个正脸特征，找到对应的多角度特征（同一人）
        # 正样本对：同一batch内的同一人
        # 负样本对：同一batch内的不同人
        
        # 计算温度缩放后的相似度
        scaled_sim = similarity / self.temperature  # [B, B]
        
        # 创建正样本mask（对角线上的配对，即同一batch内的同一人）
        # 但这里需要匹配：front_features[i] 应该与 angle_features[i] 匹配（如果是正样本对）
        # 我们需要使用pair_label来判断，但这里只有labels（person_id）
        # 所以我们需要假设batch内的配对是：front_features[i] 对应 angle_features[i]
        
        # 对于每个样本，正样本是同一人的配对，负样本是不同人的配对
        # 简化：使用对角线作为正样本（假设batch内front[i]对应angle[i]）
        positive_sim = torch.diag(scaled_sim)  # [B]
        
        # 计算InfoNCE损失：-log(exp(positive) / (exp(positive) + sum(exp(negative))))
        # 对于每个样本，正样本是它自己，负样本是batch内的其他样本
        logits = scaled_sim  # [B, B]
        labels_diag = torch.arange(len(front_features), device=front_features.device)  # [B]
        
        # 使用交叉熵损失（InfoNCE的标准形式）
        positive_loss = F.cross_entropy(logits, labels_diag)
        
        # 负样本损失（推远不同人的特征）
        negative_mask = 1 - positive_mask
        negative_sim = similarity * negative_mask
        # 使用margin loss：max(0, margin - similarity)
        negative_loss = torch.clamp(self.margin - negative_sim, min=0.0)
        negative_loss = (negative_loss * negative_mask).sum() / (negative_mask.sum() + 1e-8)
        
        return positive_loss + 0.1 * negative_loss

