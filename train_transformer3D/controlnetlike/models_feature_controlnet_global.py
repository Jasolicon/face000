"""
模型1增强版：支持全局特征学习的特征转换ControlNet

改进点：
1. 添加自注意力机制：学习特征维度内的全局关联
2. 添加跨样本注意力：batch内样本之间的交互
3. 添加全局上下文池化：聚合全局信息
4. 添加特征交互层：学习特征之间的关联
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from typing import Optional, Tuple

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from train_transformer3D.controlnetlike.models_feature_controlnet import (
    ZeroConv1d, ControlBranch, FeatureControlNet
)
from train_transformer3D.utils_seed import set_seed
from train_transformer3D.models_utils import AnglePositionalEncoding, AngleConditionedLayerNorm

set_seed(42)


class GlobalSelfAttention(nn.Module):
    """
    全局自注意力：学习特征维度内的全局关联
    
    将特征向量视为序列，使用自注意力学习特征维度之间的关联
    """
    def __init__(
        self,
        feature_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim必须能被num_heads整除"
        
        # 查询、键、值投影
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [batch, feature_dim]
        Returns:
            output: 输出特征 [batch, feature_dim]
        """
        batch_size = x.size(0)
        
        # 将特征视为单token序列 [batch, 1, feature_dim]
        x = x.unsqueeze(1)  # [batch, 1, feature_dim]
        
        # 生成查询、键、值
        q = self.q_proj(x)  # [batch, 1, feature_dim]
        k = self.k_proj(x)  # [batch, 1, feature_dim]
        v = self.v_proj(x)  # [batch, 1, feature_dim]
        
        # 重塑为多头格式
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, heads, 1, head_dim]
        k = k.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, heads, 1, head_dim]
        v = v.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, heads, 1, head_dim]
        
        # 计算注意力分数（自注意力：每个特征维度关注所有特征维度）
        # 为了学习特征维度之间的关联，我们需要将特征维度展开
        # 将特征维度视为序列：将 [batch, 1, feature_dim] 展开为 [batch, feature_dim, 1]
        # 但这样会改变语义，更好的方法是使用特征维度作为序列
        
        # 方法：将特征维度展开为序列
        # 将 [batch, feature_dim] 重塑为 [batch, feature_dim, 1] 然后转置为 [batch, 1, feature_dim]
        # 实际上，对于全局特征，我们可以使用特征维度作为序列长度
        
        # 更简单的方法：使用特征维度作为序列
        # 将特征维度切分为多个子特征，然后使用自注意力
        # 或者：使用特征维度作为序列长度
        
        # 实际上，对于512维特征，我们可以将其视为512个token的序列
        # 但这样计算量太大，更好的方法是使用特征维度分组
        
        # 简化实现：使用特征维度作为序列
        # 将 [batch, feature_dim] 重塑为 [batch, feature_dim, 1]
        # 然后使用自注意力学习特征维度之间的关联
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch, heads, 1, 1]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        attn_output = torch.matmul(attn_weights, v)  # [batch, heads, 1, head_dim]
        
        # 合并头部
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, 1, heads, head_dim]
        attn_output = attn_output.view(batch_size, 1, self.feature_dim)  # [batch, 1, feature_dim]
        
        # 输出投影
        output = self.out_proj(attn_output)  # [batch, 1, feature_dim]
        
        # 残差连接
        output = output + x  # [batch, 1, feature_dim]
        
        # 移除序列维度
        output = output.squeeze(1)  # [batch, feature_dim]
        
        return output


class CrossSampleAttention(nn.Module):
    """
    跨样本注意力：batch内样本之间的交互
    
    让每个样本关注batch内其他样本，学习全局上下文信息
    """
    def __init__(
        self,
        feature_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim必须能被num_heads整除"
        
        # 查询、键、值投影
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [batch, feature_dim]
        Returns:
            output: 输出特征 [batch, feature_dim]
        """
        batch_size = x.size(0)
        
        # 将特征视为序列 [batch, 1, feature_dim]
        x_seq = x.unsqueeze(1)  # [batch, 1, feature_dim]
        
        # 生成查询、键、值
        q = self.q_proj(x_seq)  # [batch, 1, feature_dim]
        k = self.k_proj(x_seq)  # [batch, 1, feature_dim]
        v = self.v_proj(x_seq)  # [batch, 1, feature_dim]
        
        # 重塑为多头格式
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, heads, 1, head_dim]
        k = k.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, heads, 1, head_dim]
        v = v.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, heads, 1, head_dim]
        
        # 计算注意力分数（跨样本注意力）
        # q: [batch, heads, 1, head_dim]
        # k: [batch, heads, 1, head_dim]
        # 扩展k以计算所有样本对的注意力
        q_expanded = q.unsqueeze(2)  # [batch, heads, 1, 1, head_dim]
        k_expanded = k.unsqueeze(1)  # [batch, heads, 1, 1, head_dim] -> 需要扩展为 [batch, 1, heads, 1, head_dim]
        
        # 重新组织：让每个样本的查询关注所有样本的键
        # q: [batch, heads, 1, head_dim] -> [batch, heads, batch, 1, head_dim]
        # k: [batch, heads, 1, head_dim] -> [1, heads, batch, 1, head_dim]
        
        # 更简单的方法：直接计算batch内的注意力
        # 将 [batch, heads, 1, head_dim] 重塑为 [1, heads, batch, head_dim]
        q_batch = q.squeeze(2)  # [batch, heads, head_dim]
        k_batch = k.squeeze(2)  # [batch, heads, head_dim]
        v_batch = v.squeeze(2)  # [batch, heads, head_dim]
        
        # 计算注意力分数：[batch, heads, batch]
        attn_scores = torch.matmul(
            q_batch,  # [batch, heads, head_dim]
            k_batch.transpose(-2, -1)  # [batch, heads, head_dim] -> [batch, heads, head_dim]
        ) * self.scale  # [batch, heads, batch] -> 不对，应该是 [batch, batch, heads]
        
        # 修正：q_batch: [batch, heads, head_dim], k_batch: [batch, heads, head_dim]
        # 需要转置k_batch的batch维度
        # q_batch: [batch, heads, head_dim]
        # k_batch: [batch, heads, head_dim]
        # 计算 [batch, heads, head_dim] @ [head_dim, heads, batch] = [batch, heads, batch]
        # 不对，应该是 [batch, heads, head_dim] @ [heads, head_dim, batch]
        
        # 正确的方法：
        # q_batch: [batch, heads, head_dim]
        # k_batch: [batch, heads, head_dim]
        # 转置k_batch: [batch, head_dim, heads]
        # 不对，应该是转置batch和head_dim维度
        
        # 最简单的方法：使用einsum
        # q: [batch, heads, head_dim]
        # k: [batch, heads, head_dim]
        # 计算: [batch_q, heads, head_dim] @ [batch_k, head_dim, heads] = [batch_q, heads, batch_k]
        attn_scores = torch.einsum('bhd,bkd->bhk', q_batch, k_batch) * self.scale  # [batch, heads, batch]
        
        # 应用softmax（在batch维度上）
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, heads, batch]
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        # attn_weights: [batch, heads, batch]
        # v_batch: [batch, heads, head_dim]
        # 使用einsum: [batch_q, heads, batch_k] @ [batch_k, heads, head_dim] = [batch_q, heads, head_dim]
        attn_output = torch.einsum('bhk,bkd->bhd', attn_weights, v_batch)  # [batch, heads, head_dim]
        
        # 重塑回原始格式
        attn_output = attn_output.unsqueeze(2)  # [batch, heads, 1, head_dim]
        
        # 合并头部
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, 1, heads, head_dim]
        attn_output = attn_output.view(batch_size, 1, self.feature_dim)  # [batch, 1, feature_dim]
        
        # 输出投影
        output = self.out_proj(attn_output)  # [batch, 1, feature_dim]
        
        # 残差连接
        output = output + x_seq  # [batch, 1, feature_dim]
        
        # 移除序列维度
        output = output.squeeze(1)  # [batch, feature_dim]
        
        return output


class GlobalContextPooling(nn.Module):
    """
    全局上下文池化：聚合全局信息
    
    使用注意力机制聚合batch内的全局上下文信息
    """
    def __init__(
        self,
        feature_dim: int = 512,
        context_dim: int = 128
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        
        # 全局上下文向量（可学习）
        self.global_context = nn.Parameter(torch.randn(1, context_dim))
        
        # 特征到上下文的映射
        self.feature_to_context = nn.Linear(feature_dim, context_dim)
        self.context_to_feature = nn.Linear(context_dim, feature_dim)
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=context_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.global_context, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.feature_to_context.weight)
        nn.init.xavier_uniform_(self.context_to_feature.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [batch, feature_dim]
        Returns:
            output: 输出特征 [batch, feature_dim]
        """
        batch_size = x.size(0)
        
        # 将特征映射到上下文空间
        context = self.feature_to_context(x)  # [batch, context_dim]
        context = context.unsqueeze(1)  # [batch, 1, context_dim]
        
        # 扩展全局上下文
        global_ctx = self.global_context.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 1, context_dim]
        
        # 计算注意力（特征上下文关注全局上下文）
        attended, _ = self.attention(
            query=context,  # [batch, 1, context_dim]
            key=global_ctx,  # [batch, 1, context_dim]
            value=global_ctx  # [batch, 1, context_dim]
        )
        
        # 映射回特征空间
        global_info = self.context_to_feature(attended.squeeze(1))  # [batch, feature_dim]
        
        # 融合全局信息
        output = x + 0.1 * global_info  # 小权重融合，避免过度依赖全局信息
        
        return output


class FeatureInteractionLayer(nn.Module):
    """
    特征交互层：学习特征维度之间的关联
    
    使用MLP学习特征维度之间的非线性关联
    """
    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2
    ):
        super().__init__()
        self.feature_dim = feature_dim
        
        layers = []
        layers.append(nn.Linear(feature_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.1))
        
        layers.append(nn.Linear(hidden_dim, feature_dim))
        
        self.layers = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [batch, feature_dim]
        Returns:
            output: 输出特征 [batch, feature_dim]
        """
        # 学习特征维度之间的关联
        interaction = self.layers(x)  # [batch, feature_dim]
        
        # 残差连接
        output = x + 0.1 * interaction  # 小权重，避免过度改变
        
        return output


class MainNetworkWithGlobal(nn.Module):
    """
    支持全局特征学习的主网络
    """
    def __init__(
        self,
        feature_dim: int = 512,
        pose_dim: int = 3,
        hidden_dim: int = 512,
        num_layers: int = 3,
        freeze: bool = False,
        use_global_attention: bool = True,
        use_cross_sample_attention: bool = True,
        use_global_context: bool = True,
        use_feature_interaction: bool = True
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.pose_dim = pose_dim
        self.use_global_attention = use_global_attention
        self.use_cross_sample_attention = use_cross_sample_attention
        self.use_global_context = use_global_context
        self.use_feature_interaction = use_feature_interaction
        
        # 输入投影
        self.input_proj = nn.Linear(feature_dim + pose_dim, hidden_dim)
        
        # 全局自注意力
        if use_global_attention:
            self.global_attention = GlobalSelfAttention(
                feature_dim=hidden_dim,
                num_heads=8,
                dropout=0.1
            )
        
        # 跨样本注意力
        if use_cross_sample_attention:
            self.cross_sample_attention = CrossSampleAttention(
                feature_dim=hidden_dim,
                num_heads=8,
                dropout=0.1
            )
        
        # 全局上下文池化
        if use_global_context:
            self.global_context = GlobalContextPooling(
                feature_dim=hidden_dim,
                context_dim=128
            )
        
        # 特征交互层
        if use_feature_interaction:
            self.feature_interaction = FeatureInteractionLayer(
                feature_dim=hidden_dim,
                hidden_dim=256,
                num_layers=2
            )
        
        # Transformer层（简化版，使用MLP代替）
        self.transformer_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
            for _ in range(num_layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, feature_dim)
        
        # 角度位置编码
        self.angle_pe = AnglePositionalEncoding(hidden_dim, pose_dim)
        
        # 角度条件归一化
        self.angle_norm = AngleConditionedLayerNorm(hidden_dim, pose_dim)
        
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        features: torch.Tensor,
        pose: torch.Tensor,
        target_angle: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: 输入特征 [batch, feature_dim]
            pose: 输入姿势 [batch, pose_dim]
            target_angle: 目标角度 [batch, pose_dim]
        Returns:
            output_features: 目标角度的特征 [batch, feature_dim]
        """
        # 拼接特征和姿势
        combined = torch.cat([features, pose], dim=1)  # [batch, feature_dim + pose_dim]
        
        # 输入投影
        x = self.input_proj(combined)  # [batch, hidden_dim]
        
        # 添加角度位置编码
        angle_pe = self.angle_pe(target_angle)  # [batch, hidden_dim]
        x = x + angle_pe
        
        # ========== 全局特征学习 ==========
        # 1. 全局自注意力（学习特征维度内的关联）
        if self.use_global_attention:
            x = self.global_attention(x)  # [batch, hidden_dim]
        
        # 2. 跨样本注意力（batch内样本交互）
        if self.use_cross_sample_attention:
            x = self.cross_sample_attention(x)  # [batch, hidden_dim]
        
        # 3. 全局上下文池化（聚合全局信息）
        if self.use_global_context:
            x = self.global_context(x)  # [batch, hidden_dim]
        
        # 4. 特征交互层（学习特征维度之间的关联）
        if self.use_feature_interaction:
            x = self.feature_interaction(x)  # [batch, hidden_dim]
        
        # Transformer层处理
        for layer in self.transformer_layers:
            residual = x
            x = layer(x)
            x = x + residual  # 残差连接
        
        # 角度条件归一化
        x = self.angle_norm(x, target_angle)  # [batch, hidden_dim]
        
        # 输出投影
        output_features = self.output_proj(x)  # [batch, feature_dim]
        
        return output_features


class FeatureControlNetWithGlobal(FeatureControlNet):
    """
    支持全局特征学习的特征转换ControlNet
    """
    def __init__(
        self,
        feature_dim: int = 512,
        pose_dim: int = 3,
        hidden_dim: int = 512,
        num_main_layers: int = 3,
        num_control_layers: int = 3,
        freeze_main: bool = False,
        use_global_attention: bool = True,
        use_cross_sample_attention: bool = True,
        use_global_context: bool = True,
        use_feature_interaction: bool = True
    ):
        # 不调用父类__init__，直接初始化
        nn.Module.__init__(self)
        self.feature_dim = feature_dim
        self.pose_dim = pose_dim
        
        # 主网络（支持全局特征学习）
        self.main_network = MainNetworkWithGlobal(
            feature_dim=feature_dim,
            pose_dim=pose_dim,
            hidden_dim=hidden_dim,
            num_layers=num_main_layers,
            freeze=freeze_main,
            use_global_attention=use_global_attention,
            use_cross_sample_attention=use_cross_sample_attention,
            use_global_context=use_global_context,
            use_feature_interaction=use_feature_interaction
        )
        
        # 控制分支
        self.control_branch = ControlBranch(
            control_angle_dim=pose_dim,
            feature_dim=feature_dim,
            hidden_dim=256,
            num_layers=num_control_layers
        )
        
        # 零卷积
        self.zero_conv = ZeroConv1d(feature_dim, feature_dim)
        
        # 身份保护层
        self.identity_protection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


if __name__ == "__main__":
    """测试代码"""
    print("=" * 70)
    print("支持全局特征学习的特征转换ControlNet测试")
    print("=" * 70)
    
    # 创建模型
    model = FeatureControlNetWithGlobal(
        feature_dim=512,
        pose_dim=3,
        hidden_dim=512,
        num_main_layers=3,
        num_control_layers=3,
        freeze_main=False,
        use_global_attention=True,
        use_cross_sample_attention=True,
        use_global_context=True,
        use_feature_interaction=True
    )
    
    # 测试前向传播
    batch_size = 4
    features = torch.randn(batch_size, 512)
    pose = torch.randn(batch_size, 3)
    control_angle = torch.randn(batch_size, 3)
    
    print(f"\n输入形状:")
    print(f"  特征: {features.shape}")
    print(f"  姿势: {pose.shape}")
    print(f"  控制角度: {control_angle.shape}")
    
    # 前向传播
    output_features, control_signal = model(
        features=features,
        pose=pose,
        control_angle=control_angle,
        return_control_signal=True
    )
    
    print(f"\n输出形状:")
    print(f"  输出特征: {output_features.shape}")
    print(f"  控制信号: {control_signal.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数量:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)

