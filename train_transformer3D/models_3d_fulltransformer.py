"""
3D增强的完整Transformer模型（编码器-解码器架构）
将角度和姿态作为位置编码，构造完整的Transformer网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from typing import Optional
import math

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from train_transformer3D.models_utils import AnglePositionalEncoding, AngleConditionedLayerNorm
from train_transformer3D.utils_seed import set_seed

# 设置随机种子
set_seed(42)


class PoseAndAnglePositionalEncoding(nn.Module):
    """
    姿态和角度位置编码
    将姿态（欧拉角）和角度信息编码为位置编码
    """
    def __init__(self, d_model: int, pose_dim: int = 3, angle_dim: int = 3):
        """
        初始化姿态和角度位置编码
        
        Args:
            d_model: 模型维度
            pose_dim: 姿态维度（默认3，欧拉角）
            angle_dim: 角度维度（默认3，兼容性）
        """
        super(PoseAndAnglePositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.pose_dim = pose_dim
        
        # 姿态位置编码（使用正弦/余弦编码）
        # 注意：输入维度是 pose_dim * 2（因为 sin 和 cos 拼接）
        self.pose_pe = nn.Sequential(
            nn.Linear(pose_dim * 2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # 可学习的缩放因子
        self.scale = nn.Parameter(torch.ones(1))
        
    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        """
        将姿态转换为位置编码
        
        Args:
            pose: 姿态张量 [batch_size, pose_dim] 或 [batch_size, seq_len, pose_dim]
            
        Returns:
            pos_encoding: 位置编码 [batch_size, d_model] 或 [batch_size, seq_len, d_model]
        """
        # 使用正弦/余弦编码姿态（更稳定的编码方式）
        # 将姿态角度转换为正弦和余弦
        # 注意：pose是欧拉角（度），需要转换为弧度
        if len(pose.shape) == 2:
            # [batch, pose_dim] -> [batch, pose_dim*2]
            # 将角度（度）转换为弧度
            pose_rad = pose * math.pi / 180.0  # [batch, 3]
            pose_sin = torch.sin(pose_rad)
            pose_cos = torch.cos(pose_rad)
            pose_encoded = torch.cat([pose_sin, pose_cos], dim=-1)  # [batch, pose_dim*2]
        else:
            # [batch, seq_len, pose_dim] -> [batch, seq_len, pose_dim*2]
            pose_rad = pose * math.pi / 180.0
            pose_sin = torch.sin(pose_rad)
            pose_cos = torch.cos(pose_rad)
            pose_encoded = torch.cat([pose_sin, pose_cos], dim=-1)
        
        # 投影到d_model维度
        pos_encoding = self.pose_pe(pose_encoded)
        
        # 应用缩放
        pos_encoding = pos_encoding * self.scale
        
        return pos_encoding


class TransformerEncoderDecoder3D(nn.Module):
    """
    完整的Transformer编码器-解码器架构
    编码器：处理侧面特征和3D关键点
    解码器：使用角度和姿态作为位置编码，生成正面特征
    """
    def __init__(
        self,
        d_model: int = 512,  # InsightFace特征维度：512
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        num_keypoints: int = 5,  # 3D关键点数量
        pose_dim: int = 3,  # 姿态维度（欧拉角）
        use_pose_pe: bool = True,  # 使用姿态位置编码
        use_angle_conditioning: bool = True  # 使用角度条件归一化
    ):
        """
        初始化完整的Transformer编码器-解码器
        
        Args:
            d_model: 模型维度
            nhead: 注意力头数
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比率
            activation: 激活函数
            num_keypoints: 3D关键点数量（默认5）
            pose_dim: 姿态维度（默认3，欧拉角）
            use_pose_pe: 是否使用姿态位置编码
            use_angle_conditioning: 是否使用角度条件归一化
        """
        super(TransformerEncoderDecoder3D, self).__init__()
        
        self.d_model = d_model
        self.num_keypoints = num_keypoints
        self.pose_dim = pose_dim
        self.use_pose_pe = use_pose_pe
        self.use_angle_conditioning = use_angle_conditioning
        
        # ========== 编码器部分 ==========
        # 输入特征投影
        self.input_projection = nn.Linear(d_model, d_model)
        
        # 注意：不再使用关键点编码器，只使用姿态作为位置编码
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True  # Pre-norm架构（更稳定）
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # ========== 解码器部分 ==========
        # 解码器输入投影（正面特征的目标嵌入）
        self.decoder_input_projection = nn.Linear(d_model, d_model)
        
        # 姿态和角度位置编码
        if use_pose_pe:
            self.pose_pe = PoseAndAnglePositionalEncoding(d_model, pose_dim, pose_dim)
            # 也保留原有的角度位置编码作为备选
            self.angle_pe = AnglePositionalEncoding(d_model, angle_dim=pose_dim)
        
        # 角度条件归一化
        if use_angle_conditioning:
            self.angle_conditioned_norm = AngleConditionedLayerNorm(d_model, pose_dim)
        
        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True  # Pre-norm架构
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # 输出投影层
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
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
        angles: torch.Tensor,  # 原始角度（兼容性）[batch, ...]
        keypoints_3d: torch.Tensor = None,  # 3D关键点（已废弃，不再使用）[batch, num_kp, 3]
        pose: torch.Tensor = None,  # 姿态向量 [batch, pose_dim]
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        return_residual: bool = True
    ):
        """
        前向传播
        
        Args:
            src: 侧面特征 [batch, d_model]
            angles: 原始角度（兼容性，实际使用pose）
            keypoints_3d: 3D关键点（已废弃，不再使用）[batch, num_kp, 3]
            pose: 姿态向量 [batch, pose_dim] (欧拉角)，如果为None则使用angles
            src_mask: 编码器掩码（可选）
            tgt_mask: 解码器掩码（可选）
            return_residual: 是否返回残差（True）或完整特征（False）
            
        Returns:
            output: 残差 [batch, d_model] 或完整特征 [batch, d_model]
        """
        batch_size = src.size(0)
        
        # 如果pose为None，使用angles（兼容性）
        if pose is None:
            pose = angles if angles.shape[-1] == 3 else angles[:, :3]
        
        # ========== 编码器部分 ==========
        # 1. 编码侧面特征（只使用特征，不使用关键点）
        src_features = self.input_projection(src)  # [batch, d_model]
        src_features = src_features.unsqueeze(1)  # [batch, 1, d_model]
        
        # 2. Transformer编码（只编码侧面特征）
        # 注意：TransformerEncoder使用mask参数，不是src_mask
        encoder_output = self.encoder(src_features, mask=src_mask)  # [batch, 1, d_model]
        
        # 提取编码后的侧面特征
        encoded_src = encoder_output[:, 0, :]  # [batch, d_model]
        
        # 使用编码后的特征作为memory
        memory = encoder_output  # [batch, 1, d_model]
        
        # ========== 解码器部分 ==========
        # 1. 准备解码器输入（使用编码后的侧面特征）
        decoder_input = self.decoder_input_projection(encoded_src)  # [batch, d_model]
        decoder_input = decoder_input.unsqueeze(1)  # [batch, 1, d_model]
        
        # 2. 添加姿态和角度位置编码
        if self.use_pose_pe:
            # 使用姿态作为位置编码
            pose_pe = self.pose_pe(pose)  # [batch, d_model]
            pose_pe = pose_pe.unsqueeze(1)  # [batch, 1, d_model]
            decoder_input = decoder_input + pose_pe
            
            # 也添加角度位置编码（增强）
            angle_pe = self.angle_pe(pose)  # [batch, d_model]
            angle_pe = angle_pe.unsqueeze(1)  # [batch, 1, d_model]
            decoder_input = decoder_input + angle_pe
        
        # 3. 角度条件归一化
        if self.use_angle_conditioning:
            decoder_input = decoder_input.squeeze(1)  # [batch, d_model]
            decoder_input = self.angle_conditioned_norm(decoder_input, pose)
            decoder_input = decoder_input.unsqueeze(1)  # [batch, 1, d_model]
        
        # 4. Dropout
        decoder_input = self.dropout(decoder_input)
        
        # 5. Transformer解码
        # memory: 编码器输出（只包含侧面特征）
        # tgt: 解码器输入（带位置编码的侧面特征）
        decoder_output = self.decoder(
            tgt=decoder_input,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
        )  # [batch, 1, d_model]
        
        decoder_output = decoder_output.squeeze(1)  # [batch, d_model]
        
        # 6. 输出投影
        residual = self.output_projection(decoder_output)  # [batch, d_model]
        
        if return_residual:
            return residual
        else:
            return src + residual


if __name__ == "__main__":
    # 测试模型
    print("=" * 70)
    print("测试 TransformerEncoderDecoder3D 模型")
    print("=" * 70)
    
    # 创建模型
    print("\n创建 TransformerEncoderDecoder3D 模型...")
    model = TransformerEncoderDecoder3D(
        d_model=512,  # InsightFace特征维度
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        num_keypoints=5,  # InsightFace的5个关键点
        pose_dim=3,  # 欧拉角
        use_pose_pe=True,
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
    from train_transformer3D.models_3d import TransformerDecoderOnly3D
    decoder_model = TransformerDecoderOnly3D(
        d_model=512,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        num_keypoints=5,
        pose_dim=3
    )
    decoder_params = sum(p.numel() for p in decoder_model.parameters())
    encoder_decoder_params = sum(p.numel() for p in model.parameters())
    
    print(f"TransformerDecoderOnly3D: {decoder_params:,}")
    print(f"TransformerEncoderDecoder3D: {encoder_decoder_params:,}")
    print(f"差异: {encoder_decoder_params - decoder_params:,} ({(encoder_decoder_params - decoder_params) / decoder_params * 100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
