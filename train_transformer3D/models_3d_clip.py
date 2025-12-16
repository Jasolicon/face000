"""
3D增强的Transformer模型（CLIP版本）
使用CLIP编码姿态信息，替代简单的MLP编码
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from typing import Optional

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from train_transformer3D.models_utils import AnglePositionalEncoding, AngleConditionedLayerNorm
from train_transformer3D.utils_seed import set_seed

# 设置随机种子
set_seed(42)


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


class TransformerDecoderOnly3D_CLIP(nn.Module):
    """
    CLIP增强版：融合CLIP姿态编码和Transformer解码器
    使用CLIP将姿态角度转换为文本描述，获得语义丰富的姿态表示
    """
    def __init__(
        self,
        d_model: int = 512,  # InsightFace特征维度：512
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        num_keypoints: int = 5,  # 3D关键点数量（InsightFace是5个，但CLIP版本不使用）
        pose_dim: int = 3,  # 姿态维度（欧拉角3维）
        use_pose_attention: bool = True,
        use_angle_pe: bool = True,
        use_angle_conditioning: bool = True,
        use_clip_pose_encoder: bool = True,  # 是否使用CLIP编码姿态
        device: str = 'cuda'
    ):
        """
        初始化CLIP增强的Transformer解码器
        
        Args:
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: 解码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比率
            activation: 激活函数
            num_keypoints: 3D关键点数量（保留兼容性，但CLIP版本不使用）
            pose_dim: 姿态维度（默认3，欧拉角）
            use_pose_attention: 是否使用姿态条件注意力
            use_angle_pe: 是否使用角度位置编码
            use_angle_conditioning: 是否使用角度条件归一化
            use_clip_pose_encoder: 是否使用CLIP编码姿态
            device: 设备
        """
        super(TransformerDecoderOnly3D_CLIP, self).__init__()
        
        self.d_model = d_model
        self.use_pose_attention = use_pose_attention
        self.use_angle_pe = use_angle_pe
        self.use_angle_conditioning = use_angle_conditioning
        self.use_clip_pose_encoder = use_clip_pose_encoder
        
        # ========== 姿态信息处理分支（使用CLIP） ==========
        if use_pose_attention:
            self.pose_attention = PoseConditionedAttention(d_model, d_model)  # 注意：pose_dim改为d_model，因为CLIP输出是d_model
        
        # 姿态编码器（使用CLIP或MLP）
        if use_clip_pose_encoder:
            try:
                from train_transformer3D.pose_encoder_clip import CLIPPoseEncoder
                self.pose_encoder = CLIPPoseEncoder(
                    output_dim=d_model,
                    device=device,
                    freeze_clip=True  # 冻结CLIP，只训练投影层
                )
                print(f"✓ 使用CLIP姿态编码器 (输出维度: {d_model})")
            except (ImportError, AttributeError, Exception) as e:
                print(f"⚠ 警告: 无法使用CLIP编码器 ({type(e).__name__}: {e})")
                print(f"   将使用默认MLP编码器")
                self.pose_encoder = nn.Sequential(
                    nn.Linear(pose_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, d_model)
                )
                self.use_clip_pose_encoder = False
        else:
            # 默认MLP编码器
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
        
        # ========== 第一阶段：姿态编码与融合（使用CLIP） ==========
        # 1. 编码姿态（使用CLIP或MLP）
        pose_features = self.pose_encoder(pose)  # [batch, d_model]
        
        # 2. 初始特征投影
        src_features = self.input_projection(src)  # [batch, d_model]
        
        # 3. 姿态条件注意力（如果启用）
        # 注意：如果使用CLIP，pose_features已经是d_model维度，可以直接使用
        if self.use_pose_attention:
            # 如果使用CLIP，pose_features已经是d_model维度
            if self.use_clip_pose_encoder:
                src_features, pose_attn = self.pose_attention(src_features, pose_features)
            else:
                # MLP编码器输出也是d_model，但需要从pose维度转换
                src_features, pose_attn = self.pose_attention(src_features, pose_features)
        
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
    print("测试 TransformerDecoderOnly3D_CLIP 模型")
    print("=" * 70)
    
    # 创建模型
    print("\n创建 TransformerDecoderOnly3D_CLIP 模型...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TransformerDecoderOnly3D_CLIP(
        d_model=512,  # InsightFace特征维度
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        num_keypoints=5,  # InsightFace的5个关键点（保留兼容性）
        pose_dim=3,  # 欧拉角
        use_pose_attention=True,
        use_angle_pe=True,
        use_angle_conditioning=True,
        use_clip_pose_encoder=True,  # 使用CLIP编码
        device=device
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
    from train_transformer3D.models_3d import TransformerDecoderOnly3D
    decoder3d_model = TransformerDecoderOnly3D(
        d_model=512,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        num_keypoints=5,
        pose_dim=3,
        use_pose_attention=True,
        use_angle_pe=True,
        use_angle_conditioning=True
    )
    decoder3d_params = sum(p.numel() for p in decoder3d_model.parameters())
    clip_params = sum(p.numel() for p in model.parameters())
    
    print(f"TransformerDecoderOnly3D (MLP): {decoder3d_params:,}")
    print(f"TransformerDecoderOnly3D_CLIP (CLIP): {clip_params:,}")
    print(f"差异: {clip_params - decoder3d_params:,} ({(clip_params - decoder3d_params) / decoder3d_params * 100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
