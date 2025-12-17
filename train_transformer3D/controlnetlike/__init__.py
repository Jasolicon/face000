"""
ControlNet风格的模型模块
实现两个模型：
1. FeatureControlNet: 特征转换模型（姿势+特征 -> 受角度控制 -> 该角度特征）
2. ImageControlNet: 图像生成模型（图片 -> 受姿势控制 -> 该角度图片）
"""
from .models_feature_controlnet import FeatureControlNet
from .models_image_controlnet import ImageControlNet
from .dataset_feature import FeatureControlDataset
from .dataset_image import ImageControlDataset

__all__ = [
    'FeatureControlNet',
    'ImageControlNet',
    'FeatureControlDataset',
    'ImageControlDataset',
]
