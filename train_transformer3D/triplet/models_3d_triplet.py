"""
3D增强的Transformer模型 - 三元组损失版本
专门为三元组损失优化，输出身份特征用于跨角度身份识别

核心改进：
1. 输出归一化的特征向量（用于三元组损失）
2. 增强身份区分能力
3. 保持角度不变性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from typing import Optional, Tuple

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from train_transformer3D.triplet.models_utils import (
    AnglePositionalEncoding, 
    AngleConditionedLayerNorm,
    PoseConditionedAttention,
    set_seed
)

# 设置随机种子
set_seed(42)


class IdentityProjectionHead(nn.Module):
    """
    身份投影头：将Transformer输出投影到身份特征空间
    用于三元组损失训练
    """
    def __init__(self, d_model: int = 512, identity_dim: int = 512):
        """
        Args:
            d_model: 输入特征维度
            identity_dim: 身份特征维度（通常等于d_model）
        """
        super().__init__()
        self.identity_dim = identity_dim
        
        # 多层投影，增强身份区分能力
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, identity_dim),
            nn.BatchNorm1d(identity_dim)
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
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        将Transformer输出投影到身份特征空间
        
        Args:
            features: Transformer输出 [batch, d_model]
            
        Returns:
            identity_features: 归一化的身份特征 [batch, identity_dim]
        """
        # 投影
        identity_features = self.projection(features)
        
        # L2归一化（用于余弦相似度计算）
        identity_features = F.normalize(identity_features, p=2, dim=1)
        
        return identity_features


class TransformerDecoderOnly3D_Triplet(nn.Module):
    """
    三元组损失版本的3D增强Transformer解码器
    
    主要特点：
    1. 输出归一化的身份特征（用于三元组损失）
    2. 增强身份区分能力
    3. 保持角度不变性（同一身份不同角度应该相似）
    """
    def __init__(
        self,
        d_model: int = 512,  # InsightFace特征维度：512
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        num_keypoints: int = 5,  # 3D关键点数量（InsightFace是5个）
        pose_dim: int = 3,  # 姿态维度（欧拉角3维）
        use_spatial_attention: bool = False,  # 三元组版本不使用空间注意力
        use_pose_attention: bool = True,
        use_angle_pe: bool = True,
        use_angle_conditioning: bool = True,
        identity_dim: int = 512,  # 身份特征维度
        return_identity_features: bool = True  # 是否返回身份特征（用于三元组损失）
    ):
        """
        初始化三元组损失版本的3D增强Transformer解码器
        
        Args:
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: 解码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比率
            activation: 激活函数
            num_keypoints: 3D关键点数量（默认5，InsightFace）
            pose_dim: 姿态维度（默认3，欧拉角）
            use_spatial_attention: 是否使用空间注意力融合（三元组版本默认False）
            use_pose_attention: 是否使用姿态条件注意力
            use_angle_pe: 是否使用角度位置编码
            use_angle_conditioning: 是否使用角度条件归一化
            identity_dim: 身份特征维度
            return_identity_features: 是否返回身份特征（True用于三元组损失，False用于传统损失）
        """
        super(TransformerDecoderOnly3D_Triplet, self).__init__()
        
        self.d_model = d_model
        self.identity_dim = identity_dim
        self.return_identity_features = return_identity_features
        self.use_pose_attention = use_pose_attention
        self.use_angle_pe = use_angle_pe
        self.use_angle_conditioning = use_angle_conditioning
        
        # ========== 姿态信息处理分支 ==========
        if use_pose_attention:
            self.pose_attention = PoseConditionedAttention(d_model, pose_dim)
        
        # 姿态编码器（用于位置编码）
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, 128),
            nn.ReLU(),
            nn.Linear(128, d_model)
        )
        
        # ========== 原网络结构（基础架构） ==========
        # 输入特征投影
        self.input_projection = nn.Linear(d_model, d_model)
        
        # 角度位置编码
        if use_angle_pe:
            self.angle_pe = AnglePositionalEncoding(d_model, angle_dim=pose_dim)
        
        # 角度条件归一化
        if use_angle_conditioning:
            self.angle_conditioned_norm = AngleConditionedLayerNorm(d_model, pose_dim)
        
        # 角度特征投影层（将角度编码为特征，作为memory）
        self.angle_memory_projection = nn.Linear(pose_dim, d_model)
        
        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # 输出投影层（用于生成残差）
        self.output_projection = nn.Linear(d_model, d_model)
        
        # ========== 身份特征投影头（三元组损失专用） ==========
        self.identity_head = IdentityProjectionHead(d_model, identity_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        src: torch.Tensor,  # 侧面特征 [batch, d_model]
        angles: torch.Tensor,  # 原始角度（保留兼容性）[batch, ...]
        keypoints_3d: torch.Tensor = None,  # 3D关键点（已废弃，不再使用）[batch, num_kp, 3]
        pose: torch.Tensor = None,  # 姿态向量 [batch, pose_dim] (欧拉角)
        src_mask: Optional[torch.Tensor] = None,
        return_residual: bool = False  # 三元组版本默认返回身份特征
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            src: 输入特征 [batch, d_model]
            angles: 原始角度（保留兼容性，实际使用pose）
            keypoints_3d: 3D关键点（已废弃，不再使用，保留以兼容旧代码）
            pose: 姿态向量 [batch, pose_dim] (欧拉角)，如果为None则使用angles
            src_mask: 源序列掩码（可选）
            return_residual: 是否返回残差（False时返回身份特征，True时返回残差）
            
        Returns:
            如果 return_identity_features=True 且 return_residual=False:
                identity_features: 归一化的身份特征 [batch, identity_dim]
                residual: None
            如果 return_residual=True:
                residual: 残差 [batch, d_model]
                identity_features: None
            如果 return_identity_features=True 且 return_residual=True:
                identity_features: 归一化的身份特征 [batch, identity_dim]
                residual: 残差 [batch, d_model]
        """
        batch_size = src.size(0)
        
        # 如果pose为None，使用angles（兼容性）
        if pose is None:
            pose = angles if angles.shape[-1] == 3 else angles[:, :3]
        
        # ========== 第一阶段：姿态编码与融合 ==========
        # 1. 编码姿态（作为位置编码）
        pose_features = self.pose_encoder(pose)  # [batch, d_model]
        
        # 2. 初始特征投影
        src_features = self.input_projection(src)  # [batch, d_model]
        
        # 3. 姿态条件注意力（如果启用）
        if self.use_pose_attention:
            src_features, pose_attn = self.pose_attention(src_features, pose)
        
        # 4. 融合特征和姿态编码
        # 使用可学习的加权融合（三元组版本：更强调身份信息，减少姿态影响）
        combined_features = 0.8 * src_features + 0.2 * pose_features
        
        # ========== 第二阶段：原有的角度条件处理 ==========
        # 添加角度位置编码（现在使用更精确的pose作为角度）
        if self.use_angle_pe:
            angle_pe = self.angle_pe(pose)  # pose作为角度输入
            combined_features = combined_features + angle_pe
        
        # 角度条件归一化
        if self.use_angle_conditioning:
            combined_features = self.angle_conditioned_norm(combined_features, pose)
        
        # Dropout
        combined_features = self.dropout(combined_features)
        
        # ========== 第三阶段：Transformer解码 ==========
        # 准备输入序列
        tgt = combined_features.unsqueeze(1)  # [batch, 1, d_model]
        
        # 准备memory：使用姿态编码作为memory（保留原设计思想）
        angle_memory = self.angle_memory_projection(pose)  # [batch, d_model]
        memory = angle_memory.unsqueeze(1)  # [batch, 1, d_model]
        
        # Transformer解码
        decoder_output = self.transformer_decoder(tgt, memory, tgt_mask=src_mask)  # [batch, 1, d_model]
        decoder_output = decoder_output.squeeze(1)  # [batch, d_model]
        
        # 输出投影（生成残差）
        residual = self.output_projection(decoder_output)
        
        # ========== 第四阶段：生成身份特征（三元组损失专用） ==========
        # 使用完整特征（原始特征 + 残差）生成身份特征
        full_features = src + residual  # [batch, d_model]
        identity_features = self.identity_head(full_features)  # [batch, identity_dim]
        
        # 根据返回模式决定输出
        if self.return_identity_features and not return_residual:
            # 只返回身份特征（用于三元组损失）
            return identity_features, None
        elif return_residual and not self.return_identity_features:
            # 只返回残差（传统模式）
            return residual, None
        else:
            # 同时返回身份特征和残差
            return identity_features, residual

