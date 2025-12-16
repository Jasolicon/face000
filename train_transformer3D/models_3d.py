"""
3D增强的Transformer模型
融合3D关键点和姿态估计的Transformer解码器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from typing import Optional

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from train_transformer.models import AnglePositionalEncoding, AngleConditionedLayerNorm
from train_transformer.utils_seed import set_seed

# 设置随机种子
set_seed(42)


class SpatialAttentionFusion(nn.Module):
    """
    空间注意力融合模块：使用3D关键点信息生成空间注意力图，调制特征
    """
    def __init__(self, feature_dim, num_keypoints):
        super().__init__()
        # 将关键点位置编码为特征
        self.keypoint_encoder = nn.Sequential(
            nn.Linear(num_keypoints * 2, 128),  # 输入是x,y坐标
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
        # 生成注意力权重
        self.attention_generator = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()  # 输出0-1的注意力权重
        )
    
    def forward(self, features, keypoints_2d):
        """
        features: [batch, feature_dim]
        keypoints_2d: [batch, num_keypoints, 2]  # 归一化的2D坐标
        """
        batch_size = features.shape[0]
        
        # 1. 编码关键点信息
        kp_flattened = keypoints_2d.reshape(batch_size, -1)  # [batch, num_kp*2]
        kp_encoded = self.keypoint_encoder(kp_flattened)  # [batch, feature_dim]
        
        # 2. 生成空间注意力（基于原始特征和关键点信息）
        combined = torch.cat([features, kp_encoded], dim=1)  # [batch, feature_dim*2]
        attention_weights = self.attention_generator(combined)  # [batch, 1]
        
        # 3. 应用注意力调制
        modulated_features = (1 - attention_weights) * features + attention_weights * kp_encoded
        
        return modulated_features, attention_weights


class PoseConditionedAttention(nn.Module):
    """
    姿态条件注意力：将头部姿态（旋转向量/欧拉角）作为键值对，
    让特征（查询）去关注相关的姿态信息
    """
    def __init__(self, feature_dim, pose_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.pose_dim = pose_dim
        
        # 将姿态编码为键和值
        self.pose_to_key = nn.Linear(pose_dim, feature_dim)
        self.pose_to_value = nn.Linear(pose_dim, feature_dim)
        
        # 将特征作为查询
        self.feature_to_query = nn.Linear(feature_dim, feature_dim)
        
        # 注意力缩放因子
        self.scale = feature_dim ** -0.5
        
    def forward(self, features, pose):
        """
        features: [batch, seq_len, feature_dim] 或 [batch, feature_dim]
        pose: [batch, pose_dim]  # 可以是欧拉角、旋转向量等
        """
        if len(features.shape) == 2:
            features = features.unsqueeze(1)  # [batch, 1, feature_dim]
        
        batch_size, seq_len, _ = features.shape
        
        # 生成查询、键、值
        queries = self.feature_to_query(features)  # [batch, seq_len, feature_dim]
        
        # 姿态作为全局的键和值（同一姿态用于所有序列位置）
        keys = self.pose_to_key(pose).unsqueeze(1)  # [batch, 1, feature_dim]
        values = self.pose_to_value(pose).unsqueeze(1)  # [batch, 1, feature_dim]
        
        # 计算注意力分数
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 应用注意力
        attended = torch.matmul(attention_weights, values)
        
        # 残差连接
        output = features + attended
        
        if seq_len == 1:
            output = output.squeeze(1)
            
        return output, attention_weights


class TransformerDecoderOnly3D(nn.Module):
    """
    增强版：融合3D关键点和姿态估计的Transformer解码器
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
        use_spatial_attention: bool = True,
        use_pose_attention: bool = True,
        use_angle_pe: bool = True,
        use_angle_conditioning: bool = True
    ):
        """
        初始化3D增强的Transformer解码器
        
        Args:
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: 解码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比率
            activation: 激活函数
            num_keypoints: 3D关键点数量（默认5，InsightFace）
            pose_dim: 姿态维度（默认3，欧拉角）
            use_spatial_attention: 是否使用空间注意力融合
            use_pose_attention: 是否使用姿态条件注意力
            use_angle_pe: 是否使用角度位置编码
            use_angle_conditioning: 是否使用角度条件归一化
        """
        super(TransformerDecoderOnly3D, self).__init__()
        
        self.d_model = d_model
        self.use_spatial_attention = False  # 禁用空间注意力（需要关键点）
        self.use_pose_attention = use_pose_attention
        self.use_angle_pe = use_angle_pe
        self.use_angle_conditioning = use_angle_conditioning
        
        # ========== 姿态信息处理分支 ==========
        # 注意：不再使用关键点，只使用姿态作为位置编码
        
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
        
        # 角度位置编码（保留，但使用姿态作为角度）
        if use_angle_pe:
            self.angle_pe = AnglePositionalEncoding(d_model, angle_dim=pose_dim)
        
        # 角度条件归一化（保留）
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
        return_residual: bool = True
    ):
        """
        前向传播
        
        Args:
            src: 输入特征 [batch, d_model]
            angles: 原始角度（保留兼容性，实际使用pose）
            keypoints_3d: 3D关键点（已废弃，不再使用，保留以兼容旧代码）
            pose: 姿态向量 [batch, pose_dim] (欧拉角)，如果为None则使用angles
            src_mask: 源序列掩码（可选）
            return_residual: 是否返回残差（True）或完整特征（False）
            
        Returns:
            output: 残差 [batch, d_model] 或完整特征 [batch, d_model]
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
        # 使用简单的加权融合（可以改为可学习权重）
        combined_features = 0.7 * src_features + 0.3 * pose_features
        
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
        
        # 输出投影
        residual = self.output_projection(decoder_output)
        
        if return_residual:
            return residual
        else:
            return src + residual


if __name__ == "__main__":
    # 测试模型
    print("=" * 70)
    print("测试 TransformerDecoderOnly3D 模型")
    print("=" * 70)
    
    # 创建模型
    print("\n创建 TransformerDecoderOnly3D 模型...")
    model = TransformerDecoderOnly3D(
        d_model=512,  # InsightFace特征维度
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        num_keypoints=5,  # InsightFace的5个关键点
        pose_dim=3,  # 欧拉角
        use_spatial_attention=True,
        use_pose_attention=True,
        use_angle_pe=True,
        use_angle_conditioning=True
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建测试数据
    batch_size = 4
    src = torch.randn(batch_size, 512)  # 输入特征（InsightFace: 512维）
    angles = torch.randn(batch_size, 3)  # 角度（兼容性，实际使用pose）
    keypoints_3d = torch.randn(batch_size, 5, 3)  # 5个3D关键点
    pose = torch.randn(batch_size, 3)  # 欧拉角
    
    print(f"\n输入特征形状: {src.shape}")
    print(f"3D关键点形状: {keypoints_3d.shape}")
    print(f"姿态向量形状: {pose.shape}")
    
    # 前向传播
    print("\n执行前向传播...")
    model.eval()
    with torch.no_grad():
        output = model(
            src=src,
            angles=angles,
            keypoints_3d=keypoints_3d,
            pose=pose,
            return_residual=True
        )
    
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # 对比参数量
    print("\n" + "-" * 70)
    print("参数量对比:")
    print("-" * 70)
    from train_transformer.models_decoder_only import TransformerDecoderOnly
    decoder_model = TransformerDecoderOnly(
        d_model=512,  # InsightFace特征维度
        nhead=8,
        num_layers=4,
        dim_feedforward=2048
    )
    decoder_params = sum(p.numel() for p in decoder_model.parameters())
    decoder3d_params = sum(p.numel() for p in model.parameters())
    
    print(f"TransformerDecoderOnly (2D): {decoder_params:,}")
    print(f"TransformerDecoderOnly3D (3D): {decoder3d_params:,}")
    print(f"差异: {decoder3d_params - decoder_params:,} ({(decoder3d_params - decoder_params) / decoder_params * 100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
