"""
通用人脸姿态不变网络
融合特征解耦、对比学习和姿态不变性的先进思想
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from train_transformer.utils_seed import set_seed

# 设置随机种子
set_seed(42)


class OrthogonalProjection(nn.Module):
    """正交投影层，强制身份和姿态特征正交"""
    def __init__(self, feat_dim=512, id_dim=256, pose_dim=128):
        super().__init__()
        self.feat_dim = feat_dim
        self.id_dim = id_dim
        self.pose_dim = pose_dim
        self.W_id = nn.Linear(feat_dim, id_dim, bias=False)
        self.W_pose = nn.Linear(feat_dim, pose_dim, bias=False)
        self.ortho_reg_weight = 0.01
        
    def forward(self, x):
        """
        Args:
            x: [batch, feat_dim]
        Returns:
            f_id: [batch, id_dim]
            f_pose: [batch, pose_dim]
        """
        f_id = self.W_id(x)
        f_pose = self.W_pose(x)
        return f_id, f_pose
    
    def ortho_loss(self):
        """正交约束损失"""
        W_id_norm = F.normalize(self.W_id.weight, dim=1)
        W_pose_norm = F.normalize(self.W_pose.weight, dim=1)
        ortho = torch.mm(W_id_norm, W_pose_norm.t())
        return torch.norm(ortho, p='fro') * self.ortho_reg_weight


class PoseAwareAttention(nn.Module):
    """姿态感知的自注意力机制（改进版：统一对Q、K、V注入姿态信息）"""
    def __init__(self, dim, num_heads=8, use_clip_pose_encoder=False, device='cuda'):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        # 标准自注意力参数
        self.qkv = nn.Linear(dim, dim * 3)
        
        # 姿态编码器（可选择使用CLIP或MLP）
        if use_clip_pose_encoder:
            try:
                from train_transformer3D.pose_encoder_clip import CLIPPoseEncoder
                self.pose_encoder = CLIPPoseEncoder(
                    output_dim=dim * 3,
                    device=device,
                    freeze_clip=True  # 冻结CLIP，只训练投影层
                )
                print(f"✓ 使用CLIP姿态编码器 (输出维度: {dim * 3})")
            except (ImportError, AttributeError, Exception) as e:
                print(f"⚠ 警告: 无法使用CLIP编码器 ({type(e).__name__}: {e})")
                print(f"   将使用默认MLP编码器")
                self.pose_encoder = nn.Sequential(
                    nn.Linear(3, 64),  # 3个姿态角
                    nn.ReLU(),
                    nn.Linear(64, dim * 3)  # 为Q、K、V分别生成 [batch, dim] 的编码
                )
        else:
            # 默认MLP编码器
            self.pose_encoder = nn.Sequential(
                nn.Linear(3, 64),  # 3个姿态角
                nn.ReLU(),
                nn.Linear(64, dim * 3)  # 为Q、K、V分别生成 [batch, dim] 的编码
            )
        
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, pose_angles):
        """
        Args:
            x: [batch, seq_len, dim]
            pose_angles: [batch, 3] (yaw, pitch, roll)
        Returns:
            out: [batch, seq_len, dim]
        """
        B, N, C = x.shape
        
        # 标准QKV计算
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 姿态编码（统一对Q、K、V注入姿态信息）
        pose_emb = self.pose_encoder(pose_angles)  # [B, dim*3]
        pose_q, pose_k, pose_v = pose_emb.chunk(3, dim=1)  # 各 [B, dim]
        
        # 将姿态编码reshape并加到Q、K、V
        pose_q = pose_q.view(B, self.num_heads, 1, self.head_dim)  # [B, heads, 1, head_dim]
        pose_k = pose_k.view(B, self.num_heads, 1, self.head_dim)
        pose_v = pose_v.view(B, self.num_heads, 1, self.head_dim)
        
        # 应用姿态偏置（广播到所有token）
        q = q + pose_q
        k = k + pose_k
        v = v + pose_v
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class PoseNormalizationLayer(nn.Module):
    """姿态感知的归一化层（类似AdaIN）"""
    def __init__(self, num_features, pose_dim=128):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        self.pose_embedding = nn.Sequential(
            nn.Linear(3, pose_dim),
            nn.ReLU(),
            nn.Linear(pose_dim, num_features * 2)
        )
        
    def forward(self, x, pose_angles):
        """
        Args:
            x: [batch, seq_len, num_features]
            pose_angles: [batch, 3]
        Returns:
            x_out: [batch, seq_len, num_features]
        """
        # 姿态相关的归一化参数
        pose_params = self.pose_embedding(pose_angles)  # [batch, num_features*2]
        gamma, beta = torch.chunk(pose_params, 2, dim=-1)  # 各 [batch, num_features]
        
        # 应用层归一化
        x_norm = self.norm(x)  # [batch, seq_len, num_features]
        
        # 姿态自适应的缩放和平移
        x_out = gamma.unsqueeze(1) * x_norm + beta.unsqueeze(1)
        return x_out


class PoseTransformerBlock(nn.Module):
    """姿态感知的Transformer块"""
    def __init__(self, dim, heads, mlp_dim, dropout=0.1, use_clip_pose_encoder=False, device='cuda'):
        super().__init__()
        self.attn = PoseAwareAttention(dim, heads, use_clip_pose_encoder=use_clip_pose_encoder, device=device)
        self.norm1 = PoseNormalizationLayer(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, pose_angles):
        """
        Args:
            x: [batch, seq_len, dim]
            pose_angles: [batch, 3]
        """
        # 姿态感知的自注意力
        x_norm = self.norm1(x, pose_angles)
        attn_out = self.attn(x_norm, pose_angles)
        x = x + self.dropout(attn_out)
        
        # MLP层
        x = x + self.mlp(self.norm2(x))
        return x


class PoseTransformerEncoder(nn.Module):
    """姿态感知的Transformer编码器"""
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.1, use_clip_pose_encoder=False, device='cuda'):
        super().__init__()
        self.layers = nn.ModuleList([
            PoseTransformerBlock(dim, heads, mlp_dim, dropout, use_clip_pose_encoder=use_clip_pose_encoder, device=device)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, pose_angles):
        """
        Args:
            x: [batch, seq_len, dim]
            pose_angles: [batch, 3]
        """
        for layer in self.layers:
            x = layer(x, pose_angles)
        return self.norm(x)


class IdentityEnhancer(nn.Module):
    """身份特征增强模块，去除姿态影响（改进版：使用清晰的门控机制）"""
    def __init__(self, id_dim, pose_dim=128):
        super().__init__()
        # 门控网络：计算姿态抑制权重
        self.gate_net = nn.Sequential(
            nn.Linear(id_dim + pose_dim, id_dim),
            nn.ReLU(),
            nn.Linear(id_dim, id_dim),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )
        # 姿态污染投影：将姿态特征投影到身份空间
        self.pose_proj = nn.Linear(pose_dim, id_dim, bias=False)
        
    def forward(self, id_features, pose_features):
        """
        Args:
            id_features: [batch, id_dim]
            pose_features: [batch, pose_dim]
        Returns:
            enhanced_id: [batch, id_dim]
        """
        # 计算姿态抑制门控权重
        concat = torch.cat([id_features, pose_features], dim=1)  # [batch, id_dim+pose_dim]
        gate = self.gate_net(concat)  # [batch, id_dim], 范围 [0, 1]
        
        # 计算姿态污染（在身份空间中的投影）
        pose_contamination = self.pose_proj(pose_features)  # [batch, id_dim]
        
        # 改进：使用更保守的策略，添加残差连接保护原始特征
        # 1. 只去除部分污染（0.3倍），而不是全部
        # 2. 使用残差连接，确保不会过度抑制身份特征
        cleaned_id = id_features - 0.3 * gate * pose_contamination
        enhanced_id = id_features + 0.5 * (cleaned_id - id_features)
        
        return enhanced_id


class UniversalFaceTransformer(nn.Module):
    """
    通用人脸姿态不变网络
    核心思想：解耦身份和姿态特征，学习通用转动模式
    
    注意：这个模型直接使用特征向量作为输入（而不是图像），
    因为我们的数据已经是提取好的InsightFace特征
    """
    
    def __init__(self, 
                 feat_dim=512,  # InsightFace特征维度
                 id_dim=256,
                 pose_dim=128,
                 num_pose_bins=36,
                 transformer_depth=4,
                 transformer_heads=8,
                 transformer_mlp_dim=1024,
                 dropout=0.1,
                 use_clip_pose_encoder=False,  # 是否使用CLIP姿态编码器
                 device='cuda'):
        super().__init__()
        
        self.feat_dim = feat_dim
        self.id_dim = id_dim
        self.pose_dim = pose_dim
        
        # 1. 特征投影（将输入特征投影到统一空间）
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. 特征解耦模块
        self.ortho_proj = OrthogonalProjection(feat_dim, id_dim, pose_dim)
        
        # 3. 姿态感知Transformer编码器
        # 将特征切分成多个token以充分利用Transformer
        self.num_tokens = 8  # 将512维切分成8个64维token
        self.token_dim = feat_dim // self.num_tokens  # 64
        assert feat_dim % self.num_tokens == 0, f"feat_dim ({feat_dim}) must be divisible by num_tokens ({self.num_tokens})"
        
        # Token投影层（如果需要将token维度投影到feat_dim）
        if self.token_dim != feat_dim:
            self.token_proj = nn.Linear(self.token_dim, feat_dim)
        else:
            self.token_proj = None
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, self.num_tokens, feat_dim))
        
        self.pose_encoder = PoseTransformerEncoder(
            dim=feat_dim,
            depth=transformer_depth,
            heads=transformer_heads,
            mlp_dim=transformer_mlp_dim,
            dropout=dropout,
            use_clip_pose_encoder=use_clip_pose_encoder,
            device=device
        )
        
        # 4. 身份增强模块
        self.id_enhancer = IdentityEnhancer(id_dim, pose_dim)
        
        # 5. 姿态估计头
        self.pose_head = nn.Sequential(
            nn.Linear(pose_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3)  # yaw, pitch, roll
        )
        
        # 6. 姿态原型记忆库（存储姿态原型）
        self.num_pose_bins = num_pose_bins
        # 先创建并归一化，再注册为buffer
        pose_prototypes = torch.randn(num_pose_bins, pose_dim)
        pose_prototypes = F.normalize(pose_prototypes, dim=1)
        self.register_buffer('pose_prototypes', pose_prototypes)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重（改进：对不同模块使用不同初始化策略）"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'pose_bias' in name or 'pose_encoder' in name or 'pose_embedding' in name:
                    # 姿态相关的小权重初始化
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                elif 'proj' in name or 'projection' in name:
                    # 投影层使用Xavier初始化
                    nn.init.xavier_uniform_(module.weight)
                else:
                    # 其他线性层使用Xavier初始化
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm不需要特殊初始化
                pass
            elif isinstance(module, nn.Parameter):
                # 参数（如位置编码）使用小权重初始化
                if 'pos_encoding' in name:
                    nn.init.normal_(module, mean=0.0, std=0.02)
    
    def forward(self, features, pose_angles=None, mode='train'):
        """
        Args:
            features: 输入特征 [batch, feat_dim] (InsightFace特征)
            pose_angles: 姿态角度 [batch, 3] (yaw, pitch, roll) 或 None
            mode: 'train' 或 'inference'
        Returns:
            根据mode返回不同输出
        """
        batch_size = features.shape[0]
        
        # 1. 特征投影
        base_features = self.feat_proj(features)  # [batch, feat_dim]
        
        # 2. 姿态感知的特征增强（改进：使用特征切分充分利用Transformer）
        # 将特征切分成多个token
        base_features_tokens = base_features.view(batch_size, self.num_tokens, self.token_dim)  # [batch, 8, 64]
        
        # 如果token_dim不等于feat_dim，需要投影
        if self.token_proj is not None:
            base_features_seq = self.token_proj(base_features_tokens)  # [batch, 8, feat_dim]
        else:
            base_features_seq = base_features_tokens  # [batch, 8, feat_dim]
        
        # 添加位置编码
        base_features_seq = base_features_seq + self.pos_encoding  # [batch, 8, feat_dim]
        
        if pose_angles is not None:
            # 训练时使用真实姿态标签
            enhanced_features_seq = self.pose_encoder(base_features_seq, pose_angles)  # [batch, 8, feat_dim]
        else:
            # 推理时：如果没有提供姿态，先粗略估计姿态
            # 先进行一次快速解耦以估计姿态
            id_features_temp, pose_features_temp = self.ortho_proj(base_features)
            est_pose_from_temp = self.pose_head(pose_features_temp)
            enhanced_features_seq = self.pose_encoder(base_features_seq, est_pose_from_temp)
        
        # 聚合token（使用平均池化）
        enhanced_features = enhanced_features_seq.mean(dim=1)  # [batch, feat_dim]
        
        # 3. 特征解耦（只解耦一次，在增强之后）
        id_features_enhanced, pose_features_enhanced = self.ortho_proj(enhanced_features)
        pose_features = pose_features_enhanced
        
        # 4. 使用姿态原型对齐（如果可用）
        if mode == 'train' and pose_angles is not None:
            # 计算与所有原型的相似度
            pose_features_norm = F.normalize(pose_features, dim=1)
            prototypes_norm = F.normalize(self.pose_prototypes, dim=1)
            similarity = torch.mm(pose_features_norm, prototypes_norm.t())  # [batch, num_prototypes]
            
            # 找到最近的原型并软对齐
            nearest_prototype_idx = similarity.argmax(dim=1)  # [batch]
            nearest_prototype = self.pose_prototypes[nearest_prototype_idx]  # [batch, pose_dim]
            
            # 软对齐（使用小的权重避免过度约束）
            pose_features = pose_features + 0.1 * (nearest_prototype - pose_features)
        
        # 5. 姿态归一化的身份特征
        id_features = self.id_enhancer(id_features_enhanced, pose_features)
        
        # 6. 姿态估计
        est_pose = self.pose_head(pose_features)
        
        if mode == 'inference':
            # 推理时只返回身份特征和估计姿态
            return {
                'id_features': F.normalize(id_features, dim=1),
                'pose_angles': est_pose,
                'pose_features': F.normalize(pose_features, dim=1)
            }
        
        # 训练时返回所有中间结果用于多任务学习
        # 注意：为了兼容性，保留id_features_raw和pose_features_raw（从增强后的特征解耦得到）
        return {
            'id_features': id_features,
            'pose_features': pose_features,
            'pose_angles': est_pose,
            'base_features': enhanced_features,  # 使用增强后的特征作为base_features
            'id_features_raw': id_features_enhanced,  # 解耦后的身份特征（未增强）
            'pose_features_raw': pose_features_enhanced  # 解耦后的姿态特征
        }
    
    def get_ortho_loss(self):
        """获取正交约束损失"""
        return self.ortho_proj.ortho_loss()


if __name__ == "__main__":
    # 测试模型
    print("=" * 70)
    print("测试 UniversalFaceTransformer 模型")
    print("=" * 70)
    
    model = UniversalFaceTransformer(
        feat_dim=512,
        id_dim=256,
        pose_dim=128,
        num_pose_bins=36,
        transformer_depth=4,
        transformer_heads=8,
        transformer_mlp_dim=1024
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    batch_size = 4
    features = torch.randn(batch_size, 512)
    pose_angles = torch.randn(batch_size, 3)
    
    print(f"\n输入特征形状: {features.shape}")
    print(f"姿态角度形状: {pose_angles.shape}")
    
    # 训练模式
    print("\n训练模式:")
    model.eval()
    with torch.no_grad():
        outputs = model(features, pose_angles, mode='train')
        print(f"  身份特征: {outputs['id_features'].shape}")
        print(f"  姿态特征: {outputs['pose_features'].shape}")
        print(f"  估计姿态: {outputs['pose_angles'].shape}")
    
    # 推理模式
    print("\n推理模式:")
    with torch.no_grad():
        outputs = model(features, pose_angles=None, mode='inference')
        print(f"  身份特征: {outputs['id_features'].shape}")
        print(f"  估计姿态: {outputs['pose_angles'].shape}")
        print(f"  姿态特征: {outputs['pose_features'].shape}")
    
    # 正交损失
    ortho_loss = model.get_ortho_loss()
    print(f"\n正交约束损失: {ortho_loss.item():.6f}")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
