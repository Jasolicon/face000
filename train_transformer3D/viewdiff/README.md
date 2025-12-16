# ViewDiff风格增强模块

参考ViewDiff的实现，为人脸角度转换任务提供以下改进：

## 🎯 核心改进

### 1. **LoRA线性层和姿态条件LoRA注意力**
- **LoRALinearLayer**: 低秩适应线性层，大幅减少可训练参数
- **PoseConditionedLoRAAttention**: 将姿态信息通过LoRA注入注意力机制

### 2. **轻量化3D投影层**
- **LightweightFaceProjectionLayer**: 基于3D关键点的2D→3D→2D投影
- 使用体素网格实现轻量级3D特征表示

### 3. **跨视角注意力机制**
- **CrossViewAttention**: 让不同视角的特征相互关注
- 支持多视角批处理工具

### 4. **先验保护训练**
- **PriorPreservationLoss**: 防止微调时丢失原始生成能力
- **PriorPreservationDataset**: 先验样本生成和缓存

### 5. **完整集成模型**
- **EnhancedTransformerDecoderOnly3D**: 集成所有改进的增强版模型
- **EnhancedTransformerWithPrior**: 带先验保护训练的包装器

## 📁 文件结构

```
viewdiff/
├── __init__.py                      # 模块初始化
├── lora_layers.py                   # LoRA线性层
├── pose_lora_attention.py          # 姿态条件LoRA注意力
├── face_projection_layer.py         # 轻量化3D投影层
├── multiview_utils.py              # 跨视角注意力和工具函数
├── prior_preservation.py           # 先验保护损失和数据集
├── enhanced_transformer_3d.py      # 完整集成模型
├── train_enhanced.py                # 训练脚本（待创建）
└── README.md                        # 本文件
```

## 🚀 使用方法

### 基础使用

```python
from train_transformer3D.viewdiff import EnhancedTransformerDecoderOnly3D

# 创建增强版模型
model = EnhancedTransformerDecoderOnly3D(
    d_model=512,
    nhead=8,
    num_layers=4,
    num_keypoints=5,
    pose_dim=3,
    use_lora_attention=True,      # 启用LoRA注意力
    use_projection_layer=True,    # 启用3D投影层
    use_cross_view=False,         # 单视角任务
    rank=4,                       # LoRA秩
    lora_alpha=1.0
)

# 前向传播
output = model(
    src=src_features,           # [batch, 512]
    angles=angles,              # [batch, 3]
    keypoints_3d=keypoints_3d,  # [batch, 5, 3]
    pose=pose                   # [batch, 3]
)
```

### 带先验保护的训练

```python
from train_transformer3D.viewdiff import (
    EnhancedTransformerDecoderOnly3D,
    EnhancedTransformerWithPrior
)

# 1. 创建基础模型（原始预训练模型）
base_model = EnhancedTransformerDecoderOnly3D(
    d_model=512,
    use_lora_attention=False,  # 基础模型不使用LoRA
    use_projection_layer=False
)

# 2. 创建增强模型
enhanced_model = EnhancedTransformerDecoderOnly3D(
    d_model=512,
    use_lora_attention=True,
    use_projection_layer=True
)

# 3. 包装带先验保护的模型
model_with_prior = EnhancedTransformerWithPrior(
    model=enhanced_model,
    base_model=base_model,
    lambda_prior=0.1  # 先验保护权重
)

# 4. 配置优化器（可单独优化LoRA参数）
all_params = enhanced_model.get_trainable_parameters()
lora_params = enhanced_model.get_lora_parameters()

optimizer = torch.optim.AdamW([
    {'params': all_params, 'lr': 1e-4},
    {'params': lora_params, 'lr': 1e-3}  # LoRA参数用更高学习率
])

# 5. 训练循环
for batch in dataloader:
    src, angles, keypoints, pose, targets = batch
    
    # 前向传播
    outputs = enhanced_model(
        src=src,
        angles=angles,
        keypoints_3d=keypoints,
        pose=pose
    )
    
    # 计算损失（带先验保护）
    def mse_loss(pred, target):
        return F.mse_loss(pred, target)
    
    loss, loss_dict = model_with_prior.compute_loss(
        inputs=src,
        conditions={'angles': angles, 'pose': pose, 'keypoints_3d': keypoints},
        targets=targets,
        original_loss_fn=mse_loss
    )
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 多视角训练

```python
# 使用跨视角注意力
model = EnhancedTransformerDecoderOnly3D(
    d_model=512,
    use_cross_view=True,  # 启用跨视角注意力
    n_views=5            # 5个视角
)

# 输入必须是n_views的倍数
# 例如：batch_size=8, n_views=5, 则实际输入40个样本
output = model(
    src=src_features,  # [40, 512] (8*5)
    angles=angles,
    multiview_input=True  # 标记为多视角输入
)
```

## 🔧 参数说明

### LoRA参数
- `rank`: LoRA秩，通常4-16，越小参数量越少但表达能力可能降低
- `lora_alpha`: LoRA缩放因子，控制适配强度，通常等于rank

### 3D投影层参数
- `voxel_resolution`: 体素网格分辨率，默认16，越大精度越高但计算成本增加

### 先验保护参数
- `lambda_prior`: 先验保护权重，通常0.1-0.5，越大越保守

## 📊 性能优势

1. **参数量减少**: LoRA注意力大幅减少可训练参数（约减少70-90%）
2. **训练速度**: 更少的参数意味着更快的训练和推理
3. **灵活性**: 可以单独启用/禁用各个模块
4. **先验保护**: 防止微调时过拟合，保持泛化能力

## ⚠️ 注意事项

1. **LoRA vs 原始注意力**: `use_lora_attention` 和 `use_pose_attention` 建议只启用一个
2. **多视角输入**: 使用跨视角注意力时，批次大小必须是 `n_views` 的倍数
3. **先验保护**: 需要提供基础模型，确保基础模型已冻结
4. **内存使用**: 3D投影层会增加内存使用，根据GPU内存调整 `voxel_resolution`

## 🔗 参考

- ViewDiff: https://github.com/...
- DreamBooth: 先验保护策略
- LoRA: Low-Rank Adaptation of Large Language Models

