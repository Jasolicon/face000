"""
轻量级特征变换模型
提供比Transformer更轻量的替代方案
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from typing import Optional

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from train_transformer.utils_seed import set_seed

# 设置随机种子
set_seed(42)


class AngleConditionedMLP(nn.Module):
    """
    角度条件MLP模型（轻量级）
    使用多层感知机 + 角度条件归一化
    参数量远小于Transformer
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: list = [512, 512, 768],  # 隐藏层维度
        output_dim: int = 768,
        use_angle_conditioning: bool = True,
        angle_dim: int = 5,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        初始化角度条件MLP模型
        
        Args:
            input_dim: 输入特征维度（默认768）
            hidden_dims: 隐藏层维度列表（默认[512, 512, 768]）
            output_dim: 输出特征维度（默认768）
            use_angle_conditioning: 是否使用角度条件归一化
            angle_dim: 角度维度
            dropout: Dropout比率
            activation: 激活函数（'relu', 'gelu', 'swish'）
        """
        super(AngleConditionedMLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_angle_conditioning = use_angle_conditioning
        self.angle_dim = angle_dim
        
        # 选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.GELU()
        
        # 角度条件归一化（如果使用）
        if use_angle_conditioning:
            from train_transformer.models import AngleConditionedLayerNorm
            self.angle_conditioned_norm = AngleConditionedLayerNorm(input_dim, angle_dim)
        
        # 构建MLP层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self.activation,
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✓ AngleConditionedMLP初始化完成")
        print(f"  参数量: {total_params:,} ({total_params * 4 / 1024 / 1024:.2f} MB)")
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        features: torch.Tensor,
        angles: Optional[torch.Tensor] = None,
        return_residual: bool = True
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 输入特征 [batch_size, input_dim]
            angles: 角度信息 [batch_size, angle_dim] (可选)
            return_residual: 是否返回残差（True）或完整特征（False）
            
        Returns:
            output: 残差 [batch_size, output_dim] 或完整特征 [batch_size, output_dim]
        """
        # 角度条件归一化（如果使用）
        if self.use_angle_conditioning and angles is not None:
            features = self.angle_conditioned_norm(features, angles)
        
        # MLP前向传播
        output = self.mlp(features)  # [batch_size, output_dim]
        
        if return_residual:
            # 返回残差
            return output
        else:
            # 返回完整特征
            return features + output


class ResidualMLP(nn.Module):
    """
    残差MLP模型（更轻量级）
    使用残差连接的MLP，参数量更小
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,  # 单个隐藏层维度
        num_layers: int = 3,  # 残差块数量
        output_dim: int = 768,
        use_angle_conditioning: bool = True,
        angle_dim: int = 5,
        dropout: float = 0.1
    ):
        """
        初始化残差MLP模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: 残差块数量
            output_dim: 输出特征维度
            use_angle_conditioning: 是否使用角度条件归一化
            angle_dim: 角度维度
            dropout: Dropout比率
        """
        super(ResidualMLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_angle_conditioning = use_angle_conditioning
        
        # 角度条件归一化（如果使用）
        if use_angle_conditioning:
            from train_transformer.models import AngleConditionedLayerNorm
            self.angle_conditioned_norm = AngleConditionedLayerNorm(input_dim, angle_dim)
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 残差块
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # 初始化权重
        self._init_weights()
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✓ ResidualMLP初始化完成")
        print(f"  参数量: {total_params:,} ({total_params * 4 / 1024 / 1024:.2f} MB)")
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.input_proj, self.output_proj] + list(self.residual_blocks):
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(
        self,
        features: torch.Tensor,
        angles: Optional[torch.Tensor] = None,
        return_residual: bool = True
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 输入特征 [batch_size, input_dim]
            angles: 角度信息 [batch_size, angle_dim] (可选)
            return_residual: 是否返回残差
            
        Returns:
            output: 残差或完整特征
        """
        # 角度条件归一化（如果使用）
        if self.use_angle_conditioning and angles is not None:
            features = self.angle_conditioned_norm(features, angles)
        
        # 输入投影
        x = self.input_proj(features)  # [batch_size, hidden_dim]
        
        # 残差块
        for block in self.residual_blocks:
            residual = x
            x = block(x)
            x = x + residual  # 残差连接
        
        # 输出投影
        output = self.output_proj(x)  # [batch_size, output_dim]
        
        if return_residual:
            return output
        else:
            return features + output


class LightweightTransformer(nn.Module):
    """
    轻量级Transformer（减少层数和维度）
    相比标准Transformer，参数量大幅减少
    """
    
    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 4,  # 减少注意力头数（从8减到4）
        num_layers: int = 2,  # 减少层数（从4减到2）
        dim_feedforward: int = 1024,  # 减少前馈网络维度（从2048减到1024）
        dropout: float = 0.1,
        use_angle_conditioning: bool = True,
        angle_dim: int = 5
    ):
        """
        初始化轻量级Transformer
        
        Args:
            d_model: 模型维度
            nhead: 注意力头数（减少）
            num_layers: 编码器层数（减少）
            dim_feedforward: 前馈网络维度（减少）
            dropout: Dropout比率
            use_angle_conditioning: 是否使用角度条件归一化
            angle_dim: 角度维度
        """
        super(LightweightTransformer, self).__init__()
        
        from train_transformer.models import SimpleTransformerEncoder
        
        self.model = SimpleTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_angle_pe=True,
            use_angle_conditioning=use_angle_conditioning,
            angle_dim=angle_dim
        )
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✓ LightweightTransformer初始化完成")
        print(f"  参数量: {total_params:,} ({total_params * 4 / 1024 / 1024:.2f} MB)")
    
    def forward(
        self,
        features: torch.Tensor,
        angles: Optional[torch.Tensor] = None,
        return_residual: bool = True
    ) -> torch.Tensor:
        """前向传播"""
        return self.model(features, angles, return_residual=return_residual)


if __name__ == "__main__":
    # 测试轻量级模型
    print("=" * 70)
    print("测试轻量级模型")
    print("=" * 70)
    
    batch_size = 4
    input_features = torch.randn(batch_size, 768)
    angles = torch.randn(batch_size, 5)
    
    # 1. AngleConditionedMLP
    print("\n" + "-" * 70)
    print("1. AngleConditionedMLP")
    print("-" * 70)
    model1 = AngleConditionedMLP(
        input_dim=768,
        hidden_dims=[512, 512, 768],
        output_dim=768,
        use_angle_conditioning=True
    )
    with torch.no_grad():
        output1 = model1(input_features, angles)
    print(f"输出形状: {output1.shape}")
    
    # 2. ResidualMLP
    print("\n" + "-" * 70)
    print("2. ResidualMLP")
    print("-" * 70)
    model2 = ResidualMLP(
        input_dim=768,
        hidden_dim=512,
        num_layers=3,
        output_dim=768,
        use_angle_conditioning=True
    )
    with torch.no_grad():
        output2 = model2(input_features, angles)
    print(f"输出形状: {output2.shape}")
    
    # 3. LightweightTransformer
    print("\n" + "-" * 70)
    print("3. LightweightTransformer")
    print("-" * 70)
    model3 = LightweightTransformer(
        d_model=768,
        nhead=4,
        num_layers=2,
        dim_feedforward=1024
    )
    with torch.no_grad():
        output3 = model3(input_features, angles)
    print(f"输出形状: {output3.shape}")
    
    # 4. 对比标准Transformer
    print("\n" + "-" * 70)
    print("4. 标准Transformer (对比)")
    print("-" * 70)
    from train_transformer.models import SimpleTransformerEncoder
    model4 = SimpleTransformerEncoder(
        d_model=768,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048
    )
    total_params4 = sum(p.numel() for p in model4.parameters())
    print(f"参数量: {total_params4:,} ({total_params4 * 4 / 1024 / 1024:.2f} MB)")
    
    print("\n" + "=" * 70)
    print("参数量对比:")
    print("=" * 70)
    print(f"AngleConditionedMLP:     {sum(p.numel() for p in model1.parameters()):,}")
    print(f"ResidualMLP:              {sum(p.numel() for p in model2.parameters()):,}")
    print(f"LightweightTransformer:   {sum(p.numel() for p in model3.parameters()):,}")
    print(f"标准Transformer:         {total_params4:,}")
    print("=" * 70)

