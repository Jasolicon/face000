"""
仅使用Transformer解码器的模型
使用自注意力机制，不需要编码器
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


class TransformerDecoderOnly(nn.Module):
    """
    仅使用Transformer解码器的模型
    使用自注意力机制，将角度信息作为memory输入
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
        use_angle_conditioning: bool = True,
        angle_dim: int = 5
    ):
        """
        初始化仅解码器Transformer模型
        
        Args:
            d_model: 模型维度（特征维度，默认768）
            nhead: 注意力头数
            num_layers: 解码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比率
            activation: 激活函数
            use_angle_pe: 是否使用角度位置编码
            use_angle_conditioning: 是否使用角度条件归一化
            angle_dim: 角度维度（关键点数量）
        """
        super(TransformerDecoderOnly, self).__init__()
        
        self.d_model = d_model
        self.use_angle_pe = use_angle_pe
        self.use_angle_conditioning = use_angle_conditioning
        
        # 输入特征投影层
        self.input_projection = nn.Linear(d_model, d_model)
        
        # 角度特征投影层（将角度编码为特征，作为memory）
        self.angle_memory_projection = nn.Linear(angle_dim, d_model)
        
        # 位置编码
        if use_angle_pe:
            self.angle_pe = AnglePositionalEncoding(d_model, angle_dim)
        else:
            from train_transformer.models import PositionalEncoding
            self.pos_encoder = PositionalEncoding(d_model)
        
        # 角度条件归一化（如果启用）
        if use_angle_conditioning:
            self.angle_conditioned_norm = AngleConditionedLayerNorm(d_model, angle_dim)
        
        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True  # 使用 [batch_size, seq_len, d_model] 格式
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
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
        
        # 准备输入序列（用于解码器）
        tgt = src.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # 准备memory（使用角度信息）
        # 将角度编码为memory特征
        angle_memory = self.angle_memory_projection(angles)  # [batch_size, d_model]
        memory = angle_memory.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # Transformer解码器（batch_first=True）
        # tgt: 输入序列（特征）
        # memory: 记忆序列（角度编码）
        decoder_output = self.transformer_decoder(tgt, memory, tgt_mask=src_mask)  # [batch_size, 1, d_model]
        
        # 移除序列维度
        decoder_output = decoder_output.squeeze(1)  # [batch_size, d_model]
        
        # 输出投影（输出残差）
        residual = self.output_projection(decoder_output)
        
        if return_residual:
            # 返回残差
            return residual
        else:
            # 返回完整特征（输入 + 残差）
            return src + residual


if __name__ == "__main__":
    # 测试模型
    print("=" * 70)
    print("测试 TransformerDecoderOnly 模型")
    print("=" * 70)
    
    # 创建模型
    print("\n创建 TransformerDecoderOnly 模型...")
    model = TransformerDecoderOnly(
        d_model=768,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        use_angle_pe=True,
        use_angle_conditioning=True,
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
    
    # 对比参数量
    print("\n" + "-" * 70)
    print("参数量对比:")
    print("-" * 70)
    from train_transformer.models import SimpleTransformerEncoder
    encoder_model = SimpleTransformerEncoder(
        d_model=768,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048
    )
    encoder_params = sum(p.numel() for p in encoder_model.parameters())
    decoder_params = sum(p.numel() for p in model.parameters())
    
    print(f"SimpleTransformerEncoder (编码器): {encoder_params:,}")
    print(f"TransformerDecoderOnly (解码器):  {decoder_params:,}")
    print(f"差异: {decoder_params - encoder_params:,} ({(decoder_params - encoder_params) / encoder_params * 100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)

