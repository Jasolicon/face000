"""
LoRA线性层 - 在原始线性层上添加低秩适应
参考ViewDiff的LoRALinearLayer实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinearLayer(nn.Module):
    """
    LoRA线性层 - 在原始线性层上添加低秩适应
    参考ViewDiff的LoRALinearLayer实现
    
    核心思想：
    - 原始权重冻结，不参与训练
    - 使用低秩分解（rank << min(in_features, out_features)）来学习适配
    - 通过alpha参数控制适配强度
    """
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            rank: LoRA秩（低秩分解的维度，通常4-16）
            alpha: LoRA缩放因子（控制适配强度）
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features
        
        # 原始线性层权重（冻结）
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
        
        # LoRA适配层（低秩分解：W = W_down @ W_up）
        # W_down: [in_features, rank]
        # W_up: [rank, out_features]
        # 参数量：in_features * rank + rank * out_features << in_features * out_features
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        
        # 初始化
        # W_down: Kaiming初始化（适合ReLU等激活函数）
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        # W_up: 零初始化（确保初始时LoRA输出为0，不影响原始行为）
        nn.init.zeros_(self.lora_up.weight)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch, ..., in_features]
            
        Returns:
            output: 输出特征 [batch, ..., out_features]
        """
        # 原始输出
        original_out = self.linear(x)
        
        # LoRA输出
        lora_out = self.lora_up(self.lora_down(x))
        
        # 合并（scale by alpha/rank）
        # 这样可以通过调整alpha来控制LoRA的影响强度
        scale = self.alpha / self.rank
        return original_out + scale * lora_out
    
    def get_trainable_parameters(self):
        """获取可训练参数（仅LoRA层）"""
        return list(self.lora_down.parameters()) + list(self.lora_up.parameters())
    
    def merge_weights(self):
        """
        合并LoRA权重到原始权重（推理时使用，减少计算）
        注意：合并后无法继续训练LoRA
        """
        with torch.no_grad():
            scale = self.alpha / self.rank
            lora_weight = self.lora_up.weight @ self.lora_down.weight
            self.linear.weight.data += scale * lora_weight.T

