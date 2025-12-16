"""
CycleGAN架构：用于人脸特征正面化
- 生成器：Transformer模型（侧面→正面）
- 判别器：区分真实和生成的正面/侧面特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from typing import Optional

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from train_transformer3D.models_3d import TransformerDecoderOnly3D
from train_transformer3D.models_3d_fulltransformer import TransformerEncoderDecoder3D
from train_transformer3D.models_angle_warping import FinalRecommendedModel
from train_transformer3D.utils_seed import set_seed

# 设置随机种子
set_seed(42)


class FeatureDiscriminator(nn.Module):
    """
    特征判别器：区分真实和生成的人脸特征
    使用PatchGAN架构，对特征的每个"patch"进行判别
    """
    
    def __init__(
        self,
        d_model: int = 512,
        hidden_dims: list = [256, 128, 64],
        use_sigmoid: bool = True,
        dropout: float = 0.3
    ):
        """
        初始化特征判别器
        
        Args:
            d_model: 特征维度（InsightFace: 512）
            hidden_dims: 隐藏层维度列表
            use_sigmoid: 是否在输出使用sigmoid（用于BCE损失）
            dropout: Dropout比率
        """
        super(FeatureDiscriminator, self).__init__()
        
        self.d_model = d_model
        self.use_sigmoid = use_sigmoid
        
        layers = []
        in_dim = d_model
        
        # 构建判别器网络
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        # 输出层：单个标量（真实/生成概率）
        layers.append(nn.Linear(in_dim, 1))
        
        self.model = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 输入特征 [batch, d_model]
            
        Returns:
            output: 判别结果 [batch, 1]（未经过sigmoid，用于BCEWithLogitsLoss）
        """
        output = self.model(features)  # [batch, 1]
        
        if self.use_sigmoid:
            output = torch.sigmoid(output)
        
        return output


class PatchFeatureDiscriminator(nn.Module):
    """
    PatchGAN风格的判别器
    将特征分成多个"patch"，对每个patch进行判别
    更细粒度的判别，通常效果更好
    """
    
    def __init__(
        self,
        d_model: int = 512,
        patch_size: int = 64,  # 每个patch的维度
        num_patches: int = 8,  # patch数量
        hidden_dim: int = 256,
        dropout: float = 0.3
    ):
        """
        初始化PatchGAN判别器
        
        Args:
            d_model: 特征维度
            patch_size: 每个patch的维度
            num_patches: patch数量（d_model应该能被patch_size整除）
            hidden_dim: 隐藏层维度
            dropout: Dropout比率
        """
        super(PatchFeatureDiscriminator, self).__init__()
        
        self.d_model = d_model
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        # 将特征分成patches
        assert d_model == patch_size * num_patches, \
            f"d_model ({d_model}) 必须等于 patch_size ({patch_size}) * num_patches ({num_patches})"
        
        # 每个patch的判别器
        self.patch_discriminators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(patch_size, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
            for _ in range(num_patches)
        ])
        
        # 聚合所有patch的判别结果
        self.aggregator = nn.Sequential(
            nn.Linear(num_patches, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 输入特征 [batch, d_model]
            
        Returns:
            output: 判别结果 [batch, 1]
        """
        batch_size = features.shape[0]
        
        # 将特征分成patches
        patches = features.view(batch_size, self.num_patches, self.patch_size)  # [batch, num_patches, patch_size]
        
        # 对每个patch进行判别
        patch_outputs = []
        for i, patch_disc in enumerate(self.patch_discriminators):
            patch_output = patch_disc(patches[:, i, :])  # [batch, 1]
            patch_outputs.append(patch_output)
        
        # 堆叠所有patch的输出
        patch_outputs = torch.stack(patch_outputs, dim=1)  # [batch, num_patches, 1]
        patch_outputs = patch_outputs.squeeze(-1)  # [batch, num_patches]
        
        # 聚合所有patch的判别结果
        final_output = self.aggregator(patch_outputs)  # [batch, 1]
        
        return final_output


class CycleGANGenerator(nn.Module):
    """
    CycleGAN生成器包装器
    将现有的Transformer模型包装为生成器
    """
    
    def __init__(
        self,
        generator_type: str = 'decoder_only',
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_keypoints: int = 5,
        pose_dim: int = 3,
        **kwargs
    ):
        """
        初始化生成器
        
        Args:
            generator_type: 生成器类型（decoder_only, encoder_decoder, angle_warping）
            其他参数：传递给对应的模型
        """
        super(CycleGANGenerator, self).__init__()
        
        self.generator_type = generator_type
        
        if generator_type == 'decoder_only':
            from train_transformer3D.models_3d import TransformerDecoderOnly3D
            self.generator = TransformerDecoderOnly3D(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                num_keypoints=num_keypoints,
                pose_dim=pose_dim,
                use_spatial_attention=kwargs.get('use_spatial_attention', False),
                use_pose_attention=kwargs.get('use_pose_attention', False),
                use_angle_pe=True,
                use_angle_conditioning=True
            )
        elif generator_type == 'encoder_decoder':
            from train_transformer3D.models_3d_fulltransformer import TransformerEncoderDecoder3D
            self.generator = TransformerEncoderDecoder3D(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=kwargs.get('num_encoder_layers', 4),
                num_decoder_layers=kwargs.get('num_decoder_layers', 4),
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                num_keypoints=num_keypoints,
                pose_dim=pose_dim,
                use_pose_pe=kwargs.get('use_pose_pe', True),
                use_angle_conditioning=True
            )
        elif generator_type == 'angle_warping':
            from train_transformer3D.models_angle_warping import FinalRecommendedModel
            self.generator = FinalRecommendedModel(
                d_model=d_model,
                hidden_dim=kwargs.get('hidden_dim', 256),
                num_basis=kwargs.get('num_basis', 32),
                use_basis=kwargs.get('use_basis', True),
                use_refinement=kwargs.get('use_refinement', True),
                use_attention_refine=kwargs.get('use_attention_refine', True),
                num_attention_layers=kwargs.get('num_attention_layers', 1)
            )
        else:
            raise ValueError(f"Unknown generator_type: {generator_type}")
    
    def forward(
        self,
        src: torch.Tensor,
        angles: torch.Tensor,
        keypoints_3d: torch.Tensor,
        pose: torch.Tensor,
        return_residual: bool = True
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            src: 输入特征 [batch, d_model]
            angles: 角度（兼容性）
            keypoints_3d: 3D关键点 [batch, num_kp, 3]
            pose: 姿态向量 [batch, pose_dim]
            return_residual: 是否返回残差
            
        Returns:
            output: 生成的特征 [batch, d_model]
        """
        return self.generator(
            src=src,
            angles=angles,
            keypoints_3d=keypoints_3d,
            pose=pose,
            return_residual=return_residual
        )


class CycleGAN(nn.Module):
    """
    CycleGAN完整架构
    - G_AB: 侧面→正面生成器
    - G_BA: 正面→侧面生成器（反向）
    - D_A: 侧面特征判别器
    - D_B: 正面特征判别器
    """
    
    def __init__(
        self,
        generator_type: str = 'decoder_only',
        discriminator_type: str = 'patch',  # 'simple' or 'patch'
        d_model: int = 512,
        **generator_kwargs
    ):
        """
        初始化CycleGAN
        
        Args:
            generator_type: 生成器类型
            discriminator_type: 判别器类型（'simple' 或 'patch'）
            d_model: 特征维度
            generator_kwargs: 传递给生成器的参数
        """
        super(CycleGAN, self).__init__()
        
        # 生成器：侧面→正面
        self.G_AB = CycleGANGenerator(
            generator_type=generator_type,
            d_model=d_model,
            **generator_kwargs
        )
        
        # 生成器：正面→侧面（反向，使用相同架构）
        self.G_BA = CycleGANGenerator(
            generator_type=generator_type,
            d_model=d_model,
            **generator_kwargs
        )
        
        # 判别器：侧面特征
        if discriminator_type == 'patch':
            # 确保d_model可以被patch_size整除
            patch_size = 64
            num_patches = d_model // patch_size
            if d_model % patch_size != 0:
                # 如果不能整除，使用简单判别器
                print(f"警告: d_model ({d_model}) 不能被 patch_size ({patch_size}) 整除，使用简单判别器")
                self.D_A = FeatureDiscriminator(d_model=d_model)
            else:
                self.D_A = PatchFeatureDiscriminator(
                    d_model=d_model,
                    patch_size=patch_size,
                    num_patches=num_patches
                )
        else:
            self.D_A = FeatureDiscriminator(d_model=d_model)
        
        # 判别器：正面特征
        if discriminator_type == 'patch':
            patch_size = 64
            num_patches = d_model // patch_size
            if d_model % patch_size != 0:
                self.D_B = FeatureDiscriminator(d_model=d_model)
            else:
                self.D_B = PatchFeatureDiscriminator(
                    d_model=d_model,
                    patch_size=patch_size,
                    num_patches=num_patches
                )
        else:
            self.D_B = FeatureDiscriminator(d_model=d_model)
    
    def forward(
        self,
        side_features: torch.Tensor,
        front_features: torch.Tensor,
        angles: torch.Tensor,
        keypoints_3d: torch.Tensor,
        pose: torch.Tensor
    ):
        """
        前向传播（用于推理）
        
        Args:
            side_features: 侧面特征 [batch, d_model]
            front_features: 正面特征 [batch, d_model]
            angles: 角度
            keypoints_3d: 3D关键点
            pose: 姿态向量
            
        Returns:
            fake_front: 生成的正面特征
            fake_side: 生成的侧面特征
            rec_side: 重建的侧面特征（循环一致性）
            rec_front: 重建的正面特征（循环一致性）
        """
        # 侧面→正面
        fake_front = self.G_AB(side_features, angles, keypoints_3d, pose, return_residual=False)
        
        # 正面→侧面
        fake_side = self.G_BA(front_features, angles, keypoints_3d, pose, return_residual=False)
        
        # 循环一致性：侧面→正面→侧面
        rec_side = self.G_BA(fake_front, angles, keypoints_3d, pose, return_residual=False)
        
        # 循环一致性：正面→侧面→正面
        rec_front = self.G_AB(fake_side, angles, keypoints_3d, pose, return_residual=False)
        
        return fake_front, fake_side, rec_side, rec_front


if __name__ == "__main__":
    # 测试CycleGAN
    print("=" * 70)
    print("测试 CycleGAN 架构")
    print("=" * 70)
    
    batch_size = 4
    d_model = 512
    
    # 创建模型
    print("\n创建 CycleGAN 模型...")
    model = CycleGAN(
        generator_type='decoder_only',
        discriminator_type='patch',
        d_model=d_model,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        num_keypoints=5,
        pose_dim=3
    )
    
    print(f"G_AB 参数量: {sum(p.numel() for p in model.G_AB.parameters()):,}")
    print(f"G_BA 参数量: {sum(p.numel() for p in model.G_BA.parameters()):,}")
    print(f"D_A 参数量: {sum(p.numel() for p in model.D_A.parameters()):,}")
    print(f"D_B 参数量: {sum(p.numel() for p in model.D_B.parameters()):,}")
    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建测试数据
    side_features = torch.randn(batch_size, d_model)
    front_features = torch.randn(batch_size, d_model)
    angles = torch.randn(batch_size, 3)
    keypoints_3d = torch.randn(batch_size, 5, 3)
    pose = torch.randn(batch_size, 3) * 30.0
    
    print(f"\n输入形状:")
    print(f"  侧面特征: {side_features.shape}")
    print(f"  正面特征: {front_features.shape}")
    print(f"  姿态: {pose.shape}")
    
    # 前向传播
    print("\n执行前向传播...")
    model.eval()
    with torch.no_grad():
        fake_front, fake_side, rec_side, rec_front = model(
            side_features, front_features, angles, keypoints_3d, pose
        )
    
    print(f"\n输出形状:")
    print(f"  生成正面: {fake_front.shape}")
    print(f"  生成侧面: {fake_side.shape}")
    print(f"  重建侧面: {rec_side.shape}")
    print(f"  重建正面: {rec_front.shape}")
    
    # 测试判别器
    print("\n测试判别器...")
    with torch.no_grad():
        d_a_real = model.D_A(side_features)
        d_a_fake = model.D_A(fake_side)
        d_b_real = model.D_B(front_features)
        d_b_fake = model.D_B(fake_front)
    
    print(f"D_A(真实侧面): {d_a_real.mean().item():.4f}")
    print(f"D_A(生成侧面): {d_a_fake.mean().item():.4f}")
    print(f"D_B(真实正面): {d_b_real.mean().item():.4f}")
    print(f"D_B(生成正面): {d_b_fake.mean().item():.4f}")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
