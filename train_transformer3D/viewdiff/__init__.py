"""
ViewDiff风格的增强模块
参考ViewDiff的实现，提供以下改进：
1. LoRA线性层和姿态条件LoRA注意力
2. 轻量化3D投影层
3. 跨视角注意力机制
4. 先验保护训练
"""
from .lora_layers import LoRALinearLayer
from .pose_lora_attention import PoseConditionedLoRAAttention
from .face_projection_layer import LightweightFaceProjectionLayer
from .multiview_utils import CrossViewAttention, expand_to_multiview, collapse_from_multiview
from .prior_preservation import PriorPreservationLoss, PriorPreservationDataset
from .enhanced_transformer_3d import EnhancedTransformerDecoderOnly3D, EnhancedTransformerWithPrior

__all__ = [
    'LoRALinearLayer',
    'PoseConditionedLoRAAttention',
    'LightweightFaceProjectionLayer',
    'CrossViewAttention',
    'expand_to_multiview',
    'collapse_from_multiview',
    'PriorPreservationLoss',
    'PriorPreservationDataset',
    'EnhancedTransformerDecoderOnly3D',
    'EnhancedTransformerWithPrior',
]

