"""
Transformer模型定义
用于将不同角度的DINOv2特征转换为正面图特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from typing import Optional
import os
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from train_transformer.utils_seed import set_seed

# 设置随机种子
set_seed(42)


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    使用球面角作为位置编码
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        初始化位置编码
        
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码
        
        Args:
            x: 输入张量 [seq_len, batch_size, d_model]
            
        Returns:
            x + pe: 添加位置编码后的张量
        """
        x = x + self.pe[:x.size(0), :]
        return x


class AnglePositionalEncoding(nn.Module):
    """
    基于球面角的位置编码
    将5个关键点的角度转换为位置编码
    """
    
    def __init__(self, d_model: int, angle_dim: int = 5):
        """
        初始化角度位置编码
        
        Args:
            d_model: 模型维度
            angle_dim: 角度维度（关键点数量，默认5）
        """
        super(AnglePositionalEncoding, self).__init__()
        
        self.angle_dim = angle_dim
        self.d_model = d_model
        
        # 将角度映射到d_model维度的线性层
        self.angle_projection = nn.Linear(angle_dim, d_model)
        
        # 可学习的缩放因子
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """
        将角度转换为位置编码
        
        Args:
            angles: 角度张量 [batch_size, angle_dim] 或 [batch_size, seq_len, angle_dim]
            
        Returns:
            pos_encoding: 位置编码 [batch_size, d_model] 或 [batch_size, seq_len, d_model]
        """
        # 投影到d_model维度
        pos_encoding = self.angle_projection(angles)  # [B, angle_dim] -> [B, d_model]
        
        # 应用缩放
        pos_encoding = pos_encoding * self.scale
        
        return pos_encoding


class TransformerFeatureEncoder(nn.Module):
    """
    Transformer特征编码器
    将不同角度的DINOv2特征转换为正面图特征
    """
    
    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_angle_pe: bool = True,
        angle_dim: int = 5
    ):
        """
        初始化Transformer模型
        
        Args:
            d_model: 模型维度（特征维度，默认768）
            nhead: 注意力头数
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比率
            activation: 激活函数
            use_angle_pe: 是否使用角度位置编码
            angle_dim: 角度维度（关键点数量）
        """
        super(TransformerFeatureEncoder, self).__init__()
        
        self.d_model = d_model
        self.use_angle_pe = use_angle_pe
        
        # 输入特征投影层（如果需要调整维度）
        self.input_projection = nn.Linear(d_model, d_model)
        
        # 位置编码
        if use_angle_pe:
            self.angle_pe = AnglePositionalEncoding(d_model, angle_dim)
        else:
            self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False  # 使用 [seq_len, batch_size, d_model] 格式
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Transformer解码器（可选，如果使用encoder-decoder架构）
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        src: torch.Tensor,
        angles: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        use_decoder: bool = False
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            src: 输入特征 [batch_size, d_model] 或 [batch_size, seq_len, d_model]
            angles: 角度位置编码 [batch_size, angle_dim] 或 [batch_size, seq_len, angle_dim]
            tgt: 目标特征（用于decoder，可选）
            src_mask: 源序列掩码（可选）
            tgt_mask: 目标序列掩码（可选）
            use_decoder: 是否使用decoder
            
        Returns:
            output: 输出特征 [batch_size, d_model] 或 [batch_size, seq_len, d_model]
        """
        # 处理输入维度
        if src.dim() == 2:
            # [batch_size, d_model] -> [1, batch_size, d_model]
            src = src.unsqueeze(0)
            seq_len = 1
        else:
            seq_len = src.size(1)
        
        batch_size = src.size(0)
        
        # 输入投影
        src = self.input_projection(src)  # [batch_size, seq_len, d_model]
        
        # 添加位置编码
        if self.use_angle_pe:
            # 使用角度位置编码
            if angles.dim() == 2:
                # [batch_size, angle_dim] -> [batch_size, 1, d_model]
                angle_pe = self.angle_pe(angles).unsqueeze(1)  # [batch_size, 1, d_model]
            else:
                # [batch_size, seq_len, angle_dim] -> [batch_size, seq_len, d_model]
                angle_pe = self.angle_pe(angles)  # [batch_size, seq_len, d_model]
            
            # 广播角度位置编码到所有序列位置
            if angle_pe.size(1) == 1 and seq_len > 1:
                angle_pe = angle_pe.repeat(1, seq_len, 1)
            
            src = src + angle_pe
        else:
            # 使用标准位置编码
            # 转换为 [seq_len, batch_size, d_model] 格式
            src = src.transpose(0, 1)
            src = self.pos_encoder(src)
            src = src.transpose(0, 1)  # 转回 [batch_size, seq_len, d_model]
        
        # Dropout
        src = self.dropout(src)
        
        # 转换为 [seq_len, batch_size, d_model] 格式（Transformer要求）
        src = src.transpose(0, 1)  # [seq_len, batch_size, d_model]
        
        # Transformer编码器
        encoder_output = self.transformer_encoder(src, mask=src_mask)  # [seq_len, batch_size, d_model]
        
        # 如果使用decoder
        if use_decoder and tgt is not None:
            # 准备目标序列
            if tgt.dim() == 2:
                tgt = tgt.unsqueeze(0)
            tgt = self.input_projection(tgt)
            
            # 添加位置编码
            if self.use_angle_pe:
                if angles.dim() == 2:
                    tgt_angle_pe = self.angle_pe(angles).unsqueeze(1)
                else:
                    tgt_angle_pe = self.angle_pe(angles)
                if tgt_angle_pe.size(1) == 1 and tgt.size(1) > 1:
                    tgt_angle_pe = tgt_angle_pe.repeat(1, tgt.size(1), 1)
                tgt = tgt + tgt_angle_pe
            else:
                tgt = tgt.transpose(0, 1)
                tgt = self.pos_encoder(tgt)
                tgt = tgt.transpose(0, 1)
            
            tgt = self.dropout(tgt)
            tgt = tgt.transpose(0, 1)  # [seq_len, batch_size, d_model]
            
            # Transformer解码器
            output = self.transformer_decoder(tgt, encoder_output, tgt_mask=tgt_mask)  # [seq_len, batch_size, d_model]
        else:
            # 只使用编码器
            output = encoder_output
        
        # 转回 [batch_size, seq_len, d_model] 格式
        output = output.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # 输出投影
        output = self.output_projection(output)
        
        # 如果输入是单个特征，返回单个特征
        if seq_len == 1:
            output = output.squeeze(1)  # [batch_size, d_model]
        
        return output


class AngleConditionedLayerNorm(nn.Module):
    """
    角度条件归一化层
    使用角度信息来调制特征的归一化，让角度更深入地影响特征变换
    """
    def __init__(self, d_model: int, angle_dim: int = 5):
        super(AngleConditionedLayerNorm, self).__init__()
        self.d_model = d_model
        self.angle_dim = angle_dim
        
        # 角度到缩放和偏移的映射
        self.angle_to_scale = nn.Sequential(
            nn.Linear(angle_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()  # 缩放因子在0-1之间
        )
        self.angle_to_shift = nn.Sequential(
            nn.Linear(angle_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # 标准LayerNorm
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [batch_size, d_model] 或 [batch_size, seq_len, d_model]
            angles: 角度 [batch_size, angle_dim] 或 [batch_size, seq_len, angle_dim]
        Returns:
            条件归一化后的特征
        """
        # 计算角度相关的缩放和偏移
        scale = self.angle_to_scale(angles)  # [batch_size, d_model] 或 [batch_size, seq_len, d_model]
        shift = self.angle_to_shift(angles)  # [batch_size, d_model] 或 [batch_size, seq_len, d_model]
        
        # 标准归一化
        x_norm = self.layer_norm(x)
        
        # 应用角度相关的缩放和偏移
        x_conditioned = x_norm * (1 + scale) + shift
        
        return x_conditioned


class SimpleTransformerEncoder(nn.Module):
    """
    简化的Transformer编码器
    只使用编码器，不使用解码器
    改进版：使用角度条件归一化，让角度更深入地影响特征变换
    """
    
    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_angle_pe: bool = True,
        use_angle_conditioning: bool = True,  # 新增：是否使用角度条件归一化
        angle_dim: int = 5
    ):
        """
        初始化简化Transformer模型
        
        Args:
            d_model: 模型维度（特征维度，默认768）
            nhead: 注意力头数
            num_layers: 编码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比率
            activation: 激活函数
            use_angle_pe: 是否使用角度位置编码
            use_angle_conditioning: 是否使用角度条件归一化（让角度更深入地影响特征）
            angle_dim: 角度维度（关键点数量）
        """
        super(SimpleTransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.use_angle_pe = use_angle_pe
        self.use_angle_conditioning = use_angle_conditioning
        
        # 输入特征投影层
        self.input_projection = nn.Linear(d_model, d_model)
        
        # 位置编码
        if use_angle_pe:
            self.angle_pe = AnglePositionalEncoding(d_model, angle_dim)
        else:
            self.pos_encoder = PositionalEncoding(d_model)
        
        # 角度条件归一化（如果启用）
        if use_angle_conditioning:
            self.angle_conditioned_norm = AngleConditionedLayerNorm(d_model, angle_dim)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True  # 使用 [batch_size, seq_len, d_model] 格式
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        src: torch.Tensor,
        angles: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        return_residual: bool = True
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            src: 输入特征 [batch_size, d_model]
            angles: 角度位置编码 [batch_size, angle_dim]
            src_mask: 源序列掩码（可选）
            return_residual: 是否返回残差（True）或完整特征（False）
            
        Returns:
            output: 残差 [batch_size, d_model] 或完整特征 [batch_size, d_model]
        """
        batch_size = src.size(0)
        
        # 输入投影
        src = self.input_projection(src)  # [batch_size, d_model]
        
        # 添加位置编码
        if self.use_angle_pe:
            # 使用角度位置编码
            angle_pe = self.angle_pe(angles)  # [batch_size, d_model]
            src = src + angle_pe
        else:
            # 使用标准位置编码（需要添加序列维度）
            src = src.unsqueeze(1)  # [batch_size, 1, d_model]
            src = src.transpose(0, 1)  # [1, batch_size, d_model]
            src = self.pos_encoder(src)
            src = src.transpose(0, 1)  # [batch_size, 1, d_model]
            src = src.squeeze(1)  # [batch_size, d_model]
        
        # 角度条件归一化（让角度更深入地影响特征）
        if self.use_angle_conditioning:
            src = self.angle_conditioned_norm(src, angles)  # [batch_size, d_model]
        
        # Dropout
        src = self.dropout(src)
        
        # 添加序列维度用于Transformer
        src = src.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # Transformer编码器（batch_first=True）
        encoder_output = self.transformer_encoder(src, mask=src_mask)  # [batch_size, 1, d_model]
        
        # 移除序列维度
        encoder_output = encoder_output.squeeze(1)  # [batch_size, d_model]
        
        # 输出投影（输出残差）
        residual = self.output_projection(encoder_output)
        
        if return_residual:
            # 返回残差
            return residual
        else:
            # 返回完整特征（输入 + 残差）
            return src + residual


if __name__ == "__main__":
    # 测试模型
    print("=" * 70)
    print("测试 Transformer 模型")
    print("=" * 70)
    
    # 创建模型
    print("\n创建 SimpleTransformerEncoder 模型...")
    model = SimpleTransformerEncoder(
        d_model=768,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        use_angle_pe=True,
        angle_dim=5
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建测试数据
    batch_size = 4
    src = torch.randn(batch_size, 768)  # 输入特征
    angles = torch.randn(batch_size, 5)  # 角度位置编码
    
    print(f"\n输入形状: {src.shape}")
    print(f"角度形状: {angles.shape}")
    
    # 前向传播
    print("\n执行前向传播...")
    model.eval()
    with torch.no_grad():
        output = model(src, angles)
    
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # 测试 TransformerFeatureEncoder
    print("\n" + "-" * 70)
    print("测试 TransformerFeatureEncoder 模型...")
    model2 = TransformerFeatureEncoder(
        d_model=768,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        use_angle_pe=True,
        angle_dim=5
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model2.parameters()):,}")
    
    with torch.no_grad():
        output2 = model2(src, angles, use_decoder=False)
    
    print(f"输出形状: {output2.shape}")
    print(f"输出范围: [{output2.min().item():.4f}, {output2.max().item():.4f}]")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)

