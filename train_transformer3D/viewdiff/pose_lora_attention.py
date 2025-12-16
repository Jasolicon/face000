"""
姿态条件LoRA注意力 - 将姿态信息通过LoRA注入注意力
参考ViewDiff的PoseCondLoRAAttnProcessor2_0
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .lora_layers import LoRALinearLayer


class PoseConditionedLoRAAttention(nn.Module):
    """
    姿态条件LoRA注意力 - 将姿态信息通过LoRA注入注意力机制
    
    核心思想：
    - 原始注意力权重冻结
    - 使用LoRA在查询、键、值投影中注入姿态信息
    - 姿态条件与特征拼接后输入LoRA层
    """
    def __init__(self, feature_dim, pose_dim, num_heads=8, rank=4, alpha=1.0):
        """
        Args:
            feature_dim: 特征维度
            pose_dim: 姿态维度（如欧拉角3维）
            num_heads: 注意力头数
            rank: LoRA秩
            alpha: LoRA缩放因子
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.pose_dim = pose_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert feature_dim % num_heads == 0, "feature_dim必须能被num_heads整除"
        
        # 原始注意力投影层（冻结）
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        # 冻结原始权重
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            layer.weight.requires_grad = False
            if layer.bias is not None:
                layer.bias.requires_grad = False
        
        # LoRA投影层（姿态条件）
        # 输入：特征 + 姿态，输出：特征
        lora_input_dim = feature_dim + pose_dim
        self.q_lora = LoRALinearLayer(lora_input_dim, feature_dim, rank, alpha)
        self.k_lora = LoRALinearLayer(lora_input_dim, feature_dim, rank, alpha)
        self.v_lora = LoRALinearLayer(lora_input_dim, feature_dim, rank, alpha)
        self.out_lora = LoRALinearLayer(feature_dim, feature_dim, rank, alpha)
        
    def forward(self, x, pose, return_attention=False):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch, seq_len, feature_dim] 或 [batch, feature_dim]
            pose: 姿态向量 [batch, pose_dim]
            return_attention: 是否返回注意力权重
            
        Returns:
            output: 输出特征 [batch, seq_len, feature_dim] 或 [batch, feature_dim]
            attention_weights: 注意力权重（如果return_attention=True）
        """
        # 处理输入维度
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch, 1, feature_dim]
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len, _ = x.shape
        
        # 扩展姿态条件到序列长度
        pose_expanded = pose.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, pose_dim]
        
        # 准备LoRA条件（特征+姿态）
        lora_cond = torch.cat([x, pose_expanded], dim=-1)  # [batch, seq_len, feature_dim + pose_dim]
        
        # 计算查询、键、值（原始 + LoRA）
        q_original = self.q_proj(x)
        q_lora = self.q_lora(lora_cond)
        q = q_original + q_lora
        
        k_original = self.k_proj(x)
        k_lora = self.k_lora(lora_cond)  # 自注意力时使用相同条件
        k = k_original + k_lora
        
        v_original = self.v_proj(x)
        v_lora = self.v_lora(lora_cond)
        v = v_original + v_lora
        
        # 多头注意力
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch, heads, seq_len, seq_len]
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 注意力输出
        attn_output = torch.matmul(attn_weights, v)  # [batch, heads, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.feature_dim)
        
        # 输出投影（原始 + LoRA）
        out_original = self.out_proj(attn_output)
        out_lora = self.out_lora(attn_output)
        output = out_original + out_lora
        
        # 如果输入是2D，输出也应该是2D
        if squeeze_output:
            output = output.squeeze(1)
        
        if return_attention:
            return output, attn_weights
        return output
    
    def get_trainable_parameters(self):
        """获取可训练参数（仅LoRA层）"""
        params = []
        params.extend(self.q_lora.get_trainable_parameters())
        params.extend(self.k_lora.get_trainable_parameters())
        params.extend(self.v_lora.get_trainable_parameters())
        params.extend(self.out_lora.get_trainable_parameters())
        return params

