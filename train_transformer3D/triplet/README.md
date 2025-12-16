# 三元组损失模块

本文件夹包含所有与三元组损失相关的代码。

## 📁 文件结构

```
triplet/
├── __init__.py                    # 模块初始化，导出所有公共接口
├── angle_aware_loss.py            # 角度感知三元组损失实现
├── dataset_triplet.py             # 三元组损失专用数据集（模仿dataset_3d.py）
├── models_3d_triplet.py          # 三元组损失版本的3D Transformer模型
├── train_3d_triplet.py           # 三元组损失训练脚本
└── README.md                      # 本文件
```

## 🎯 核心模块

### 1. `angle_aware_loss.py`
- **AngleAwareTripletSampler**: 角度感知三元组采样器
- **AngleAwareTripletLoss**: 角度感知三元组损失函数

### 2. `dataset_triplet.py`
- **TripletFaceDataset3D**: 三元组损失专用数据集
- **triplet_collate_fn**: 三元组损失专用的collate函数
- **create_triplet_dataloader**: 创建数据加载器
- **create_triplet_train_val_test_dataloaders**: 创建训练/验证/测试数据加载器

### 3. `models_3d_triplet.py`
- **IdentityProjectionHead**: 身份投影头
- **TransformerDecoderOnly3D_Triplet**: 三元组损失版本的3D Transformer模型

### 4. `train_3d_triplet.py`
- 完整的训练脚本，支持三元组损失训练

## 🚀 使用方法

### 导入模块

```python
# 方式1：从triplet模块导入
from train_transformer3D.triplet import (
    AngleAwareTripletLoss,
    TransformerDecoderOnly3D_Triplet,
    TripletFaceDataset3D,
    create_triplet_train_val_test_dataloaders
)

# 方式2：直接导入
from train_transformer3D.triplet.angle_aware_loss import AngleAwareTripletLoss
from train_transformer3D.triplet.models_3d_triplet import TransformerDecoderOnly3D_Triplet
from train_transformer3D.triplet.dataset_triplet import TripletFaceDataset3D
```

### 训练模型

```bash
python train_transformer3D/triplet/train_3d_triplet.py \
    --data_dir train/datas/file \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --margin 0.2 \
    --alpha 2.0 \
    --beta 1.5 \
    --angle_threshold 30.0 \
    --use_amp
```

## 📊 数据集

`TripletFaceDataset3D` 模仿 `dataset_3d.py` 的结构，但专门为三元组损失优化：

- ✅ 包含 `person_name` 字段（身份标签）
- ✅ 使用 `triplet_collate_fn` 正确处理批处理
- ✅ 支持按 `person_name` 分割数据集
- ✅ 自动处理角度过滤

## 🔧 关键特性

1. **独立模块**: 所有三元组相关代码都在此文件夹中
2. **不依赖外部**: 不依赖 `train_transformer copy` 目录
3. **完整功能**: 包含数据集、模型、损失函数和训练脚本
4. **易于使用**: 通过 `__init__.py` 提供统一的导入接口

## 📝 注意事项

- 数据集需要包含 `person_name` 字段
- 建议 `batch_size >= 16` 以确保有足够的身份进行三元组采样
- 三元组损失需要至少2个不同身份才能工作

