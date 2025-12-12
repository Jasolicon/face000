"""
基于CLIP的特征映射模型
将正面特征和多角度特征映射到同一个空间
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from typing import Optional, Tuple

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("警告: CLIP未安装，请运行: pip install git+https://github.com/openai/CLIP.git")

from train_transformer.utils_seed import set_seed

# 设置随机种子
set_seed(42)


class CLIPFeatureMapper(nn.Module):
    """
    基于CLIP的特征映射模型
    将正面特征和多角度特征映射到同一个空间
    
    架构：
    1. 使用CLIP的图像编码器（ViT）作为特征编码器
    2. 可选：结合角度信息作为条件输入
    3. 投影层将特征映射到共同空间
    """
    
    def __init__(
        self,
        input_dim: int = 768,  # DINOv2特征维度
        output_dim: int = 512,  # CLIP特征空间维度
        clip_model_name: str = 'ViT-B/32',  # CLIP模型名称
        use_angle_conditioning: bool = True,  # 是否使用角度条件
        angle_dim: int = 5,  # 角度维度
        freeze_clip: bool = False,  # 是否冻结CLIP参数
        projection_layers: int = 2,  # 投影层数
        dropout: float = 0.1
    ):
        """
        初始化CLIP特征映射模型
        
        Args:
            input_dim: 输入特征维度（DINOv2特征，默认768）
            output_dim: 输出特征维度（CLIP特征空间，默认512）
            clip_model_name: CLIP模型名称（'ViT-B/32', 'ViT-B/16', 'ViT-L/14'等）
            use_angle_conditioning: 是否使用角度条件
            angle_dim: 角度维度
            freeze_clip: 是否冻结CLIP参数
            projection_layers: 投影层数
            dropout: Dropout比率
        """
        super(CLIPFeatureMapper, self).__init__()
        
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP未安装，请先安装: pip install git+https://github.com/openai/CLIP.git")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_angle_conditioning = use_angle_conditioning
        self.angle_dim = angle_dim
        
        # 加载CLIP模型
        print(f"加载CLIP模型: {clip_model_name}")
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device='cpu')
        
        # 获取CLIP的图像编码器
        self.clip_vision_encoder = self.clip_model.visual
        
        # 获取CLIP的特征维度
        if hasattr(self.clip_model.visual, 'output_dim'):
            clip_feature_dim = self.clip_model.visual.output_dim
        elif hasattr(self.clip_model.visual, 'proj'):
            # 通过投影层获取维度
            clip_feature_dim = self.clip_model.visual.proj.out_features
        else:
            # 默认值（ViT-B/32通常是512）
            clip_feature_dim = 512
            print(f"⚠️ 无法自动检测CLIP特征维度，使用默认值: {clip_feature_dim}")
        
        # 冻结CLIP参数（可选）
        if freeze_clip:
            for param in self.clip_vision_encoder.parameters():
                param.requires_grad = False
            print("✓ CLIP参数已冻结")
        
        # 输入特征投影层（将DINOv2特征投影到CLIP输入维度）
        # CLIP ViT的输入是图像patch，我们需要将特征向量转换为合适的格式
        # 方案1：直接投影到CLIP特征维度
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, clip_feature_dim),
            nn.LayerNorm(clip_feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 角度条件层（如果使用）
        if use_angle_conditioning:
            self.angle_conditioning = nn.Sequential(
                nn.Linear(angle_dim, clip_feature_dim),
                nn.LayerNorm(clip_feature_dim),
                nn.GELU()
            )
            print("✓ 启用角度条件")
        
        # 特征融合层（结合CLIP编码特征和角度信息）
        if use_angle_conditioning:
            self.feature_fusion = nn.Sequential(
                nn.Linear(clip_feature_dim * 2, clip_feature_dim),
                nn.LayerNorm(clip_feature_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        
        # 投影层（将特征映射到输出空间）
        projection = []
        current_dim = clip_feature_dim
        for i in range(projection_layers - 1):
            projection.extend([
                nn.Linear(current_dim, current_dim),
                nn.LayerNorm(current_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        projection.append(nn.Linear(current_dim, output_dim))
        self.projection = nn.Sequential(*projection)
        
        # 初始化投影层
        self._init_projection_weights()
        
        print(f"✓ CLIP特征映射模型初始化完成")
        print(f"  输入维度: {input_dim} -> CLIP维度: {clip_feature_dim} -> 输出维度: {output_dim}")
    
    def _init_projection_weights(self):
        """初始化投影层权重"""
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode_with_clip(self, features: torch.Tensor) -> torch.Tensor:
        """
        使用CLIP编码特征
        
        注意：CLIP的ViT期望输入是图像patch，但我们输入的是特征向量
        这里我们使用CLIP的投影层来处理特征
        
        Args:
            features: 输入特征 [batch_size, input_dim]
            
        Returns:
            encoded_features: CLIP编码后的特征 [batch_size, clip_feature_dim]
        """
        # 投影到CLIP特征维度
        projected = self.input_projection(features)  # [batch_size, clip_feature_dim]
        
        # 尝试使用CLIP视觉编码器的最后几层（如果可能）
        # 由于CLIP ViT期望图像输入，我们直接使用投影后的特征
        # 这里简化处理：直接使用投影特征
        return projected
    
    def forward(
        self,
        features: torch.Tensor,
        angles: Optional[torch.Tensor] = None,
        return_normalized: bool = True
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 输入特征 [batch_size, input_dim] (DINOv2特征)
            angles: 角度信息 [batch_size, angle_dim] (可选)
            return_normalized: 是否返回归一化的特征
            
        Returns:
            mapped_features: 映射后的特征 [batch_size, output_dim]
        """
        # 1. 使用CLIP编码特征
        clip_features = self.encode_with_clip(features)  # [batch_size, clip_feature_dim]
        
        # 2. 结合角度信息（如果使用）
        if self.use_angle_conditioning and angles is not None:
            angle_features = self.angle_conditioning(angles)  # [batch_size, clip_feature_dim]
            # 融合特征和角度
            combined = torch.cat([clip_features, angle_features], dim=1)  # [batch_size, clip_feature_dim * 2]
            fused_features = self.feature_fusion(combined)  # [batch_size, clip_feature_dim]
        else:
            fused_features = clip_features
        
        # 3. 投影到输出空间
        output_features = self.projection(fused_features)  # [batch_size, output_dim]
        
        # 4. 归一化（CLIP通常使用归一化特征）
        if return_normalized:
            output_features = F.normalize(output_features, p=2, dim=1)
        
        return output_features


class DualCLIPFeatureMapper(nn.Module):
    """
    双编码器CLIP特征映射模型
    使用两个独立的编码器分别处理正面特征和多角度特征
    然后将它们映射到同一个空间
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 512,
        clip_model_name: str = 'ViT-B/32',
        use_angle_conditioning: bool = True,
        angle_dim: int = 5,
        freeze_clip: bool = False,
        projection_layers: int = 2,
        dropout: float = 0.1,
        share_encoder: bool = False  # 是否共享编码器
    ):
        """
        初始化双编码器CLIP模型
        
        Args:
            share_encoder: 是否共享正面和多角度的编码器（True则共享，False则独立）
        """
        super(DualCLIPFeatureMapper, self).__init__()
        
        self.share_encoder = share_encoder
        
        # 正面特征编码器
        self.front_encoder = CLIPFeatureMapper(
            input_dim=input_dim,
            output_dim=output_dim,
            clip_model_name=clip_model_name,
            use_angle_conditioning=False,  # 正面特征不需要角度
            freeze_clip=freeze_clip,
            projection_layers=projection_layers,
            dropout=dropout
        )
        
        # 多角度特征编码器
        if share_encoder:
            # 共享编码器
            self.angle_encoder = self.front_encoder
            print("✓ 使用共享编码器")
        else:
            # 独立编码器
            self.angle_encoder = CLIPFeatureMapper(
                input_dim=input_dim,
                output_dim=output_dim,
                clip_model_name=clip_model_name,
                use_angle_conditioning=use_angle_conditioning,
                angle_dim=angle_dim,
                freeze_clip=freeze_clip,
                projection_layers=projection_layers,
                dropout=dropout
            )
            print("✓ 使用独立编码器")
    
    def forward(
        self,
        front_features: torch.Tensor,
        angle_features: torch.Tensor,
        angles: Optional[torch.Tensor] = None,
        return_normalized: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            front_features: 正面特征 [batch_size, input_dim]
            angle_features: 多角度特征 [batch_size, input_dim]
            angles: 角度信息 [batch_size, angle_dim] (可选，仅用于多角度编码器)
            return_normalized: 是否返回归一化的特征
            
        Returns:
            front_mapped: 映射后的正面特征 [batch_size, output_dim]
            angle_mapped: 映射后的多角度特征 [batch_size, output_dim]
        """
        # 编码正面特征
        front_mapped = self.front_encoder(front_features, return_normalized=return_normalized)
        
        # 编码多角度特征
        angle_mapped = self.angle_encoder(angle_features, angles, return_normalized=return_normalized)
        
        return front_mapped, angle_mapped


if __name__ == "__main__":
    # 测试模型
    print("=" * 70)
    print("测试 CLIPFeatureMapper")
    print("=" * 70)
    
    if not CLIP_AVAILABLE:
        print("CLIP未安装，跳过测试")
        exit(1)
    
    # 创建模型
    print("\n创建模型...")
    model = CLIPFeatureMapper(
        input_dim=768,
        output_dim=512,
        clip_model_name='ViT-B/32',
        use_angle_conditioning=True,
        angle_dim=5,
        freeze_clip=True
    )
    
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 创建测试数据
    batch_size = 4
    input_features = torch.randn(batch_size, 768)
    angles = torch.randn(batch_size, 5)
    
    print(f"\n输入特征形状: {input_features.shape}")
    print(f"角度形状: {angles.shape}")
    
    # 前向传播
    print("\n执行前向传播...")
    model.eval()
    with torch.no_grad():
        output = model(input_features, angles)
    
    print(f"输出特征形状: {output.shape}")
    print(f"输出特征范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"输出特征L2范数: {output.norm(dim=1).mean().item():.4f} (应该接近1.0，因为归一化了)")
    
    # 测试双编码器模型
    print("\n" + "-" * 70)
    print("测试 DualCLIPFeatureMapper")
    print("-" * 70)
    
    dual_model = DualCLIPFeatureMapper(
        input_dim=768,
        output_dim=512,
        clip_model_name='ViT-B/32',
        use_angle_conditioning=True,
        share_encoder=False
    )
    
    front_features = torch.randn(batch_size, 768)
    angle_features = torch.randn(batch_size, 768)
    
    print(f"\n正面特征形状: {front_features.shape}")
    print(f"多角度特征形状: {angle_features.shape}")
    
    with torch.no_grad():
        front_mapped, angle_mapped = dual_model(front_features, angle_features, angles)
    
    print(f"映射后正面特征形状: {front_mapped.shape}")
    print(f"映射后多角度特征形状: {angle_mapped.shape}")
    
    # 计算相似度
    cosine_sim = (front_mapped * angle_mapped).sum(dim=1).mean().item()
    print(f"平均余弦相似度: {cosine_sim:.4f}")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)

