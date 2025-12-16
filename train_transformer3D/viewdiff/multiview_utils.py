"""
多视角批处理工具和跨视角注意力机制
参考ViewDiff的CrossFrameAttentionProcessor2_0
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def expand_to_multiview(x, n_views):
    """
    将批次展开为多视角格式
    
    Args:
        x: [batch*n_views, ...] 输入张量
        n_views: 视角数量
        
    Returns:
        output: [batch, n_views, ...] 多视角格式
    """
    batch_n_views = x.shape[0]
    batch_size = batch_n_views // n_views
    
    assert batch_n_views % n_views == 0, f"批次大小({batch_n_views})必须是视角数({n_views})的倍数"
    
    view_dims = x.shape[1:]
    return x.view(batch_size, n_views, *view_dims)


def collapse_from_multiview(x):
    """
    将多视角格式折叠回批次
    
    Args:
        x: [batch, n_views, ...] 多视角格式
        
    Returns:
        output: [batch*n_views, ...] 批次格式
    """
    batch_size, n_views = x.shape[:2]
    view_dims = x.shape[2:]
    return x.view(batch_size * n_views, *view_dims)


class CrossViewAttention(nn.Module):
    """
    跨视角注意力：让不同视角的特征相互关注
    
    核心思想：
    - 每个视角的查询关注所有视角（包括自己）的键值
    - 通过注意力机制融合多视角信息
    - 参考ViewDiff的CrossFrameAttentionProcessor2_0
    """
    def __init__(self, feature_dim, num_heads=8, n_views=5):
        """
        Args:
            feature_dim: 特征维度
            num_heads: 注意力头数
            n_views: 视角数量
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.n_views = n_views
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert feature_dim % num_heads == 0, "feature_dim必须能被num_heads整除"
        
        # 注意力投影层
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, x, return_attention=False):
        """
        前向传播
        
        Args:
            x: [batch*n_views, seq_len, feature_dim] 或 [batch*n_views, feature_dim]
            注意：批次必须是n_views的倍数
            return_attention: 是否返回注意力权重
            
        Returns:
            output: [batch*n_views, seq_len, feature_dim] 或 [batch*n_views, feature_dim]
            attention_weights: 注意力权重（如果return_attention=True）
        """
        # 确保是3D张量
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch*n_views, 1, feature_dim]
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_n_views, seq_len, _ = x.shape
        batch_size = batch_n_views // self.n_views
        
        assert batch_n_views % self.n_views == 0, f"批次大小({batch_n_views})必须是视角数({self.n_views})的倍数"
        
        # 展开为多视角格式
        x_multiview = expand_to_multiview(x, self.n_views)  # [batch, n_views, seq_len, feature_dim]
        
        # 为每个视角计算查询、键、值
        queries = self.q_proj(x_multiview)  # [batch, n_views, seq_len, feature_dim]
        keys = self.k_proj(x_multiview)  # [batch, n_views, seq_len, feature_dim]
        values = self.v_proj(x_multiview)  # [batch, n_views, seq_len, feature_dim]
        
        # 重塑为多头注意力
        batch_size, n_views, seq_len, feat_dim = queries.shape
        
        queries = queries.view(batch_size, n_views, seq_len, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, n_views, seq_len, self.num_heads, self.head_dim)
        values = values.view(batch_size, n_views, seq_len, self.num_heads, self.head_dim)
        
        # 转置维度：[batch, n_views, seq_len, heads, head_dim] -> [batch, n_views, heads, seq_len, head_dim]
        queries = queries.permute(0, 1, 3, 2, 4)  # [batch, n_views, heads, seq_len, head_dim]
        keys = keys.permute(0, 1, 3, 2, 4)  # [batch, n_views, heads, seq_len, head_dim]
        values = values.permute(0, 1, 3, 2, 4)  # [batch, n_views, heads, seq_len, head_dim]
        
        # 计算注意力分数
        # 每个视角的查询关注所有视角的键
        # queries: [batch, n_views, heads, seq_len, head_dim]
        # keys: [batch, n_views, heads, seq_len, head_dim]
        # 扩展维度以计算所有视角对的注意力
        queries_expanded = queries.unsqueeze(2)  # [batch, n_views, 1, heads, seq_len, head_dim]
        keys_expanded = keys.unsqueeze(1)  # [batch, 1, n_views, heads, seq_len, head_dim]
        
        # 计算注意力分数：[batch, n_views, n_views, heads, seq_len, seq_len]
        attn_scores = torch.matmul(
            queries_expanded,  # [batch, n_views, 1, heads, seq_len, head_dim]
            keys_expanded.transpose(-1, -2)  # [batch, 1, n_views, heads, head_dim, seq_len]
        ) * self.scale  # [batch, n_views, n_views, heads, seq_len, seq_len]
        
        # 应用softmax（在n_views维度上，让每个视角关注所有视角）
        attn_weights = F.softmax(attn_scores.mean(dim=-1), dim=2)  # [batch, n_views, n_views, heads, seq_len]
        
        # 应用注意力权重
        values_expanded = values.unsqueeze(1)  # [batch, 1, n_views, heads, seq_len, head_dim]
        attn_weights_expanded = attn_weights.unsqueeze(-1)  # [batch, n_views, n_views, heads, seq_len, 1]
        
        attn_output = torch.sum(
            attn_weights_expanded * values_expanded,
            dim=2  # 在n_views维度上聚合
        )  # [batch, n_views, heads, seq_len, head_dim]
        
        # 合并头部
        attn_output = attn_output.permute(0, 1, 3, 2, 4).contiguous()  # [batch, n_views, seq_len, heads, head_dim]
        attn_output = attn_output.view(batch_size, n_views, seq_len, feat_dim)
        
        # 输出投影
        output = self.out_proj(attn_output)  # [batch, n_views, seq_len, feature_dim]
        
        # 折叠回原始批次维度
        output = collapse_from_multiview(output)  # [batch*n_views, seq_len, feature_dim]
        
        # 如果输入是2D，输出也应该是2D
        if squeeze_output:
            output = output.squeeze(1)  # [batch*n_views, feature_dim]
        
        if return_attention:
            return output, attn_weights
        return output

