"""
角度条件特征变换网络（Angle-Conditioned Feature Warping）
核心思想：将角度变化建模为特征空间的仿射变换，用神经网络学习这个变换矩阵
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

from train_transformer3D.utils_seed import set_seed

# 设置随机种子
set_seed(42)


class AngleConditionedFeatureWarping(nn.Module):
    """
    基于角度的人脸特征变换网络
    核心：学习从任意角度到正面的特征空间变换
    """
    
    def __init__(
        self,
        d_model: int = 512,  # InsightFace特征维度：512
        hidden_dim: int = 256,
        num_basis: int = 32,  # 特征基向量数量
        use_basis: bool = True,  # 是否使用特征基重建
        use_refinement: bool = True  # 是否使用残差细化
    ):
        """
        初始化角度条件特征变换网络
        
        Args:
            d_model: 模型维度（特征维度）
            hidden_dim: 隐藏层维度
            num_basis: 特征基向量数量
            use_basis: 是否使用特征基重建
            use_refinement: 是否使用残差细化
        """
        super(AngleConditionedFeatureWarping, self).__init__()
        
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.use_basis = use_basis
        self.use_refinement = use_refinement
        
        # 1. 角度到变换矩阵的生成器（核心创新）
        # 输出：缩放参数 + 平移参数
        self.angle_to_transform = nn.Sequential(
            nn.Linear(3, 128),           # 输入：欧拉角[yaw, pitch, roll]
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, d_model * 2)  # 输出：缩放+平移参数 [batch, d_model*2]
        )
        
        # 2. 可学习的特征基（basis）
        # 假设正面人脸特征可以用一组基向量的线性组合表示
        if use_basis:
            self.feature_basis = nn.Parameter(
                torch.randn(num_basis, d_model) * 0.02  # 32个基向量，每个512维
            )
            
            # 3. 注意力引导的特征选择
            self.attention_selector = nn.Sequential(
                nn.Linear(d_model + 3, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_basis),          # 选择哪些基向量重要
                nn.Softmax(dim=-1)
            )
        
        # 4. 残差细化网络
        if use_refinement:
            self.refinement = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, d_model)
            )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        if self.use_basis:
            # 特征基使用较小的初始化
            nn.init.normal_(self.feature_basis, mean=0.0, std=0.02)
    
    def forward(
        self,
        side_features: torch.Tensor,  # 侧面特征 [batch, d_model]
        angles: torch.Tensor,  # 当前角度 [batch, 3] (欧拉角)
        return_residual: bool = True
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            side_features: 侧面特征 [batch, d_model]
            angles: 当前角度 [batch, 3] (欧拉角，单位：度)
            return_residual: 是否返回残差（True）或完整特征（False）
            
        Returns:
            output: 残差 [batch, d_model] 或完整特征 [batch, d_model]
        """
        batch_size = side_features.shape[0]
        
        # 1. 生成角度特定的变换参数
        transform_params = self.angle_to_transform(angles)  # [batch, d_model*2]
        scale = transform_params[:, :self.d_model].sigmoid() * 2.0  # [batch, d_model], 0-2范围
        translation = transform_params[:, self.d_model:]  # [batch, d_model]
        
        # 2. 应用仿射变换：特征空间warping
        # 核心公式：frontal ≈ scale ⊙ side_features + translation
        warped_features = side_features * scale + translation
        
        # 3. 基于正面特征基重建（可选）
        if self.use_basis:
            # 计算每个基向量的重要性权重
            attention_input = torch.cat([warped_features, angles], dim=1)  # [batch, d_model+3]
            basis_weights = self.attention_selector(attention_input)  # [batch, num_basis]
            
            # 重建正面特征
            basis_features = torch.einsum('bk,kd->bd', 
                                         basis_weights, 
                                         self.feature_basis)  # [batch, d_model]
            
            # 融合：变换特征 + 基重建特征
            # 可学习的融合权重（通过门控机制）
            fusion_gate = torch.sigmoid(
                torch.sum(warped_features * basis_features, dim=1, keepdim=True)
            )  # [batch, 1]
            
            combined = fusion_gate * warped_features + (1 - fusion_gate) * basis_features
        else:
            combined = warped_features
        
        # 4. 残差细化（可选）
        if self.use_refinement:
            residual = self.refinement(combined)
            # 使用较小的残差权重，保持稳定性
            refined_features = combined + 0.1 * residual
        else:
            refined_features = combined
        
        # 5. 返回残差或完整特征
        if return_residual:
            return refined_features - side_features  # 返回残差
        else:
            return refined_features  # 返回完整特征


class FrequencyEncoder(nn.Module):
    """
    NeRF风格频率编码
    将角度编码为高频特征，增强模型对角度变化的敏感性
    """
    
    def __init__(self, num_freq: int = 10):
        """
        初始化频率编码器
        
        Args:
            num_freq: 频率数量
        """
        super(FrequencyEncoder, self).__init__()
        self.num_freq = num_freq
        # 频率带：2^0, 2^1, ..., 2^(num_freq-1)
        self.register_buffer('freq_bands', 2.0 ** torch.arange(num_freq))
        
    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """
        将角度编码为高频特征
        
        Args:
            angles: 角度张量 [batch, 3] (欧拉角，单位：度)
            
        Returns:
            encoded: 编码后的角度 [batch, 3 * num_freq * 2]
        """
        # 转换为弧度
        angles_rad = angles * math.pi / 180.0  # [batch, 3]
        angles_rad = angles_rad.unsqueeze(-1)  # [batch, 3, 1]
        
        # 扩频
        scaled = angles_rad * self.freq_bands.unsqueeze(0).unsqueeze(0)  # [batch, 3, num_freq]
        
        # 正弦余弦编码
        sin_enc = torch.sin(scaled)
        cos_enc = torch.cos(scaled)
        
        # 拼接
        encoded = torch.cat([sin_enc, cos_enc], dim=-1)  # [batch, 3, num_freq*2]
        
        return encoded.flatten(1)  # [batch, 3*num_freq*2]


class NeRFLikeFeatureField(nn.Module):
    """
    受NeRF启发的特征场模型
    将人脸特征建模为角度连续函数
    """
    
    def __init__(
        self,
        d_model: int = 512,
        hidden_dim: int = 256,
        num_freq: int = 10,
        use_view_dependent: bool = True
    ):
        """
        初始化NeRF风格特征场
        
        Args:
            d_model: 模型维度
            hidden_dim: 隐藏层维度
            num_freq: 频率编码的频率数量
            use_view_dependent: 是否使用视角依赖调制
        """
        super(NeRFLikeFeatureField, self).__init__()
        
        self.d_model = d_model
        self.use_view_dependent = use_view_dependent
        
        # 频率编码（类似NeRF的positional encoding）
        self.freq_encoder = FrequencyEncoder(num_freq=num_freq)
        
        # 特征场MLP：输入角度，输出特征修正
        encoded_dim = 3 * num_freq * 2  # 编码后的角度维度
        self.field_mlp = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, d_model)  # 输出特征修正量
        )
        
        # 视角依赖的调制（可选）
        if use_view_dependent:
            self.view_dependent_modulation = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Linear(64, d_model)
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        side_features: torch.Tensor,  # 侧面特征 [batch, d_model]
        query_angle: torch.Tensor,  # 想要生成的角度（如正面[0,0,0]）[batch, 3]
        view_angle: torch.Tensor,  # 输入特征的角度 [batch, 3]
        return_residual: bool = True
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            side_features: 侧面特征 [batch, d_model]
            query_angle: 想要生成的角度（如正面[0,0,0]）[batch, 3]
            view_angle: 输入特征的角度 [batch, 3]
            return_residual: 是否返回残差
            
        Returns:
            output: 残差或完整特征 [batch, d_model]
        """
        # 1. 编码查询角度（我们想要正面）
        query_encoded = self.freq_encoder(query_angle)  # [batch, 3*num_freq*2]
        
        # 2. 通过特征场获取修正
        feature_correction = self.field_mlp(query_encoded)  # [batch, d_model]
        
        # 3. 视角依赖调制（可选）
        if self.use_view_dependent:
            view_modulation = self.view_dependent_modulation(view_angle)  # [batch, d_model]
            # 应用调制
            feature_correction = feature_correction * torch.sigmoid(view_modulation)
        
        # 4. 生成目标角度特征
        # 公式：frontal = side + f(query_angle) * g(view_angle)
        frontal_features = side_features + feature_correction
        
        if return_residual:
            return frontal_features - side_features
        else:
            return frontal_features


class SelfAttentionBlock(nn.Module):
    """自注意力精修块"""
    
    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        """
        初始化自注意力块
        
        Args:
            d_model: 模型维度
            nhead: 注意力头数
            dropout: Dropout比率
        """
        super(SelfAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            d_model, 
            num_heads=nhead, 
            batch_first=True,
            dropout=dropout
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch, d_model]
            
        Returns:
            output: 精修后的特征 [batch, d_model]
        """
        # x: [batch, d_model]
        x_expanded = x.unsqueeze(1)  # [batch, 1, d_model]
        attended, _ = self.attention(x_expanded, x_expanded, x_expanded)
        attended = attended.squeeze(1)  # [batch, d_model]
        
        # 残差连接和归一化
        output = self.norm(x + self.dropout(attended))
        return output


class FinalRecommendedModel(nn.Module):
    """
    最终推荐模型：角度条件特征变换 + 轻量自注意力精修
    简单、高效、理论上合理
    """
    
    def __init__(
        self,
        d_model: int = 512,
        hidden_dim: int = 256,
        num_basis: int = 32,
        use_basis: bool = True,
        use_refinement: bool = True,
        use_attention_refine: bool = True,
        num_attention_layers: int = 1
    ):
        """
        初始化最终推荐模型
        
        Args:
            d_model: 模型维度
            hidden_dim: 隐藏层维度
            num_basis: 特征基向量数量
            use_basis: 是否使用特征基重建
            use_refinement: 是否使用残差细化
            use_attention_refine: 是否使用自注意力精修
            num_attention_layers: 自注意力层数
        """
        super(FinalRecommendedModel, self).__init__()
        
        self.d_model = d_model
        
        # 核心：角度条件特征变换
        self.angle_warping = AngleConditionedFeatureWarping(
            d_model=d_model,
            hidden_dim=hidden_dim,
            num_basis=num_basis,
            use_basis=use_basis,
            use_refinement=use_refinement
        )
        
        # 轻量自注意力精修（可选）
        if use_attention_refine:
            self.attention_refiners = nn.ModuleList([
                SelfAttentionBlock(d_model)
                for _ in range(num_attention_layers)
            ])
        else:
            self.attention_refiners = None
        
        # 输出投影（可选）
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1)
        )
    
    def forward(
        self,
        src: torch.Tensor,  # 侧面特征 [batch, d_model]
        angles: torch.Tensor,  # 当前角度 [batch, 3] (兼容性，实际使用pose)
        keypoints_3d: torch.Tensor,  # 3D关键点 [batch, num_kp, 3] (兼容性，未使用)
        pose: torch.Tensor,  # 姿态向量 [batch, pose_dim] (欧拉角)
        src_mask: Optional[torch.Tensor] = None,  # 兼容性
        tgt_mask: Optional[torch.Tensor] = None,  # 兼容性
        return_residual: bool = True
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            src: 侧面特征 [batch, d_model]
            angles: 原始角度（兼容性，实际使用pose）
            keypoints_3d: 3D关键点（兼容性，未使用）
            pose: 姿态向量 [batch, pose_dim] (欧拉角)
            src_mask: 编码器掩码（兼容性，未使用）
            tgt_mask: 解码器掩码（兼容性，未使用）
            return_residual: 是否返回残差
            
        Returns:
            output: 残差或完整特征 [batch, d_model]
        """
        # 使用pose作为角度（欧拉角）
        # 角度条件变换
        warped = self.angle_warping(src, pose, return_residual=False)  # [batch, d_model]
        
        # 自注意力精修（可选）
        if self.attention_refiners is not None:
            refined = warped
            for attention_refiner in self.attention_refiners:
                refined = attention_refiner(refined)
        else:
            refined = warped
        
        # 输出投影
        output = self.output_projection(refined)
        
        # 返回残差或完整特征
        if return_residual:
            return output - src
        else:
            return src + output


if __name__ == "__main__":
    # 测试模型
    print("=" * 70)
    print("测试角度条件特征变换模型")
    print("=" * 70)
    
    # 创建模型
    print("\n创建 FinalRecommendedModel 模型...")
    model = FinalRecommendedModel(
        d_model=512,
        hidden_dim=256,
        num_basis=32,
        use_basis=True,
        use_refinement=True,
        use_attention_refine=True,
        num_attention_layers=1
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建测试数据
    batch_size = 4
    src = torch.randn(batch_size, 512)  # 输入特征（InsightFace: 512维）
    angles = torch.randn(batch_size, 3)  # 角度（兼容性）
    keypoints_3d = torch.randn(batch_size, 5, 3)  # 5个3D关键点（兼容性）
    pose = torch.randn(batch_size, 3) * 30.0  # 欧拉角（±30度范围）
    
    print(f"\n输入特征形状: {src.shape}")
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
    angle_warping_params = sum(p.numel() for p in model.parameters())
    
    print(f"TransformerDecoderOnly3D: {decoder_params:,}")
    print(f"FinalRecommendedModel: {angle_warping_params:,}")
    print(f"差异: {angle_warping_params - decoder_params:,} ({(angle_warping_params - decoder_params) / decoder_params * 100:.1f}%)")
    print(f"参数量减少: {decoder_params - angle_warping_params:,} ({(decoder_params - angle_warping_params) / decoder_params * 100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
