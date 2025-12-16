"""
增强版3D Transformer解码器，集成所有改进：
1. 姿态条件LoRA注意力
2. 轻量化3D投影层
3. 跨视角注意力
4. 先验保护训练支持
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from train_transformer3D.triplet.models_utils import (
    AnglePositionalEncoding,
    AngleConditionedLayerNorm,
    PoseConditionedAttention,
    set_seed
)
from .pose_lora_attention import PoseConditionedLoRAAttention
from .face_projection_layer import LightweightFaceProjectionLayer
from .multiview_utils import CrossViewAttention
from .prior_preservation import PriorPreservationLoss

# 设置随机种子
set_seed(42)


class EnhancedTransformerDecoderOnly3D(nn.Module):
    """
    增强版3D Transformer解码器，集成所有改进：
    1. 姿态条件LoRA注意力
    2. 轻量化3D投影层
    3. 跨视角注意力
    4. 先验保护训练支持
    
    参考ViewDiff的实现，但针对人脸角度转换任务优化
    """
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        activation='gelu',
        num_keypoints=5,
        pose_dim=3,
        use_lora_attention=True,
        use_projection_layer=True,
        use_cross_view=False,  # 是否使用跨视角注意力
        n_views=5,  # 视角数量（如果使用跨视角）
        rank=4,  # LoRA秩
        lora_alpha=1.0,
        use_angle_pe=True,
        use_angle_conditioning=True,
        use_pose_attention=True  # 是否使用原始姿态注意力（与LoRA注意力二选一）
    ):
        """
        初始化增强版3D Transformer解码器
        
        Args:
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: 解码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比率
            activation: 激活函数
            num_keypoints: 3D关键点数量
            pose_dim: 姿态维度
            use_lora_attention: 是否使用LoRA注意力
            use_projection_layer: 是否使用3D投影层
            use_cross_view: 是否使用跨视角注意力
            n_views: 视角数量
            rank: LoRA秩
            lora_alpha: LoRA缩放因子
            use_angle_pe: 是否使用角度位置编码
            use_angle_conditioning: 是否使用角度条件归一化
            use_pose_attention: 是否使用原始姿态注意力（如果False，只使用LoRA注意力）
        """
        super().__init__()
        
        self.d_model = d_model
        self.use_lora_attention = use_lora_attention
        self.use_projection_layer = use_projection_layer
        self.use_cross_view = use_cross_view
        self.use_pose_attention = use_pose_attention
        
        # ========== 1. 姿态条件LoRA注意力 ==========
        if use_lora_attention:
            self.pose_lora_attention = PoseConditionedLoRAAttention(
                feature_dim=d_model,
                pose_dim=pose_dim,
                num_heads=nhead,
                rank=rank,
                alpha=lora_alpha
            )
        
        # ========== 2. 轻量化3D投影层 ==========
        if use_projection_layer:
            self.face_projection = LightweightFaceProjectionLayer(
                feature_dim=d_model,
                num_keypoints=num_keypoints,
                voxel_resolution=16
            )
        
        # ========== 3. 跨视角注意力（可选） ==========
        if use_cross_view:
            self.cross_view_attention = CrossViewAttention(
                feature_dim=d_model,
                num_heads=nhead,
                n_views=n_views
            )
        
        # ========== 4. 原网络架构 ==========
        # 姿态编码器
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, 128),
            nn.ReLU(),
            nn.Linear(128, d_model)
        )
        
        # 输入投影
        self.input_projection = nn.Linear(d_model, d_model)
        
        # 原始姿态条件注意力（可选，与LoRA注意力二选一）
        if use_pose_attention and not use_lora_attention:
            self.pose_attention = PoseConditionedAttention(d_model, pose_dim)
        
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
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, d_model)
        
        # Dropout
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
        keypoints_3d: torch.Tensor = None,  # 3D关键点 [batch, num_kp, 3]
        pose: torch.Tensor = None,  # 姿态向量 [batch, pose_dim] (欧拉角)
        src_mask: Optional[torch.Tensor] = None,
        return_residual: bool = True,
        multiview_input: bool = False  # 是否是多视角输入
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            src: 输入特征 [batch, d_model]
            angles: 原始角度（保留兼容性，实际使用pose）
            keypoints_3d: 3D关键点 [batch, num_kp, 3]
            pose: 姿态向量 [batch, pose_dim] (欧拉角)，如果为None则使用angles
            src_mask: 源序列掩码（可选）
            return_residual: 是否返回残差（True）或完整特征（False）
            multiview_input: 是否是多视角输入（用于跨视角注意力）
            
        Returns:
            output: 残差 [batch, d_model] 或完整特征 [batch, d_model]
        """
        batch_size = src.size(0)
        
        # 如果pose为None，使用angles（兼容性）
        if pose is None:
            pose = angles if angles.shape[-1] == 3 else angles[:, :3]
        
        # ========== 阶段1：姿态编码与融合 ==========
        # 1. 编码姿态（作为位置编码）
        pose_features = self.pose_encoder(pose)  # [batch, d_model]
        
        # 2. 初始特征投影
        src_features = self.input_projection(src)  # [batch, d_model]
        
        # 3. 姿态条件注意力
        if self.use_lora_attention:
            # 使用LoRA注意力
            src_features, _ = self.pose_lora_attention(
                src_features.unsqueeze(1),  # 添加序列维度
                pose,
                return_attention=False
            )
            src_features = src_features.squeeze(1)
        elif self.use_pose_attention:
            # 使用原始姿态注意力
            src_features, _ = self.pose_attention(src_features, pose)
        
        # 4. 融合特征和姿态编码
        # 使用可学习的加权融合（LoRA版本：更强调身份信息）
        if self.use_lora_attention:
            combined_features = 0.8 * src_features + 0.2 * pose_features
        else:
            combined_features = 0.7 * src_features + 0.3 * pose_features
        
        # ========== 阶段2：3D投影（如果提供3D关键点） ==========
        if self.use_projection_layer and keypoints_3d is not None:
            # 将特征重塑为空间格式 [batch, C, H, W]
            # 这里假设是全局特征，重塑为1x1空间
            h = w = 1
            spatial_features = combined_features.view(batch_size, self.d_model, h, w)
            
            # 应用3D投影层
            projected_features = self.face_projection(
                spatial_features,
                keypoints_3d,
                pose  # 目标姿态
            )
            
            # 重塑回特征向量
            combined_features = projected_features.view(batch_size, -1)
        
        # ========== 阶段3：跨视角注意力（如果是多视角） ==========
        if self.use_cross_view and multiview_input:
            # 假设输入已经是多视角格式 [batch*n_views, ...]
            combined_features = self.cross_view_attention(combined_features)
        
        # ========== 阶段4：原有的角度条件处理 ==========
        # 添加角度位置编码（现在使用更精确的pose作为角度）
        if hasattr(self, 'angle_pe'):
            angle_pe = self.angle_pe(pose)  # pose作为角度输入
            combined_features = combined_features + angle_pe
        
        # 角度条件归一化
        if hasattr(self, 'angle_conditioned_norm'):
            combined_features = self.angle_conditioned_norm(combined_features, pose)
        
        # Dropout
        combined_features = self.dropout(combined_features)
        
        # ========== 阶段5：Transformer解码 ==========
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
        
        if return_residual:
            return residual
        else:
            return src + residual
    
    def get_trainable_parameters(self, include_base=True):
        """
        获取可训练参数（用于优化器配置）
        
        Args:
            include_base: 是否包含基础参数（False时只返回LoRA参数）
            
        Returns:
            params: 参数列表
        """
        params = []
        
        if include_base:
            # 基础参数
            params.extend(list(self.pose_encoder.parameters()))
            params.extend(list(self.input_projection.parameters()))
            params.extend(list(self.transformer_decoder.parameters()))
            params.extend(list(self.output_projection.parameters()))
            
            if hasattr(self, 'angle_pe'):
                params.extend(list(self.angle_pe.parameters()))
            if hasattr(self, 'angle_conditioned_norm'):
                params.extend(list(self.angle_conditioned_norm.parameters()))
            if hasattr(self, 'pose_attention'):
                params.extend(list(self.pose_attention.parameters()))
            
            params.extend(list(self.angle_memory_projection.parameters()))
        
        # LoRA参数（如果使用）
        if self.use_lora_attention:
            params.extend(self.pose_lora_attention.get_trainable_parameters())
        
        # 投影层参数
        if self.use_projection_layer:
            params.extend(list(self.face_projection.parameters()))
        
        # 跨视角注意力参数
        if self.use_cross_view:
            params.extend(list(self.cross_view_attention.parameters()))
        
        return params
    
    def get_lora_parameters(self):
        """仅获取LoRA参数（用于单独优化）"""
        params = []
        if self.use_lora_attention:
            params.extend(self.pose_lora_attention.get_trainable_parameters())
        return params


class EnhancedTransformerWithPrior(nn.Module):
    """
    包装器，集成先验保护训练
    
    在训练时，同时优化原始任务损失和先验保护损失，
    防止微调时丢失原始模型的生成能力
    """
    def __init__(self, model, base_model=None, lambda_prior=0.1):
        """
        Args:
            model: 当前正在微调的模型
            base_model: 原始预训练模型（用于先验保护）
            lambda_prior: 先验保护权重
        """
        super().__init__()
        self.model = model
        self.lambda_prior = lambda_prior
        
        if base_model is not None:
            self.prior_loss = PriorPreservationLoss(base_model, lambda_prior)
        else:
            self.prior_loss = None
    
    def forward(self, *args, **kwargs):
        """前向传播（直接调用模型）"""
        return self.model(*args, **kwargs)
    
    def compute_loss(self, inputs, conditions, targets, original_loss_fn):
        """
        计算带先验保护的总损失
        
        Args:
            inputs: 输入数据
            conditions: 条件字典（包含angles, pose, keypoints_3d等）
            targets: 目标数据
            original_loss_fn: 原始任务损失函数
            
        Returns:
            total_loss: 总损失
            loss_dict: 损失详情字典
        """
        # 计算模型输出
        outputs = self.model(inputs, **conditions)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # 计算原始任务损失
        original_loss = original_loss_fn(outputs, targets)
        
        # 如果启用先验保护，添加先验损失
        if self.prior_loss is not None:
            total_loss, loss_dict = self.prior_loss(
                self.model, inputs, conditions, original_loss
            )
            return total_loss, loss_dict
        
        return original_loss, {'original_loss': original_loss.item() if torch.is_tensor(original_loss) else original_loss}

