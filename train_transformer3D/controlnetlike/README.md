# ControlNet风格模型

参考ControlNet设计，实现两个模型用于精确控制特征和图像生成，同时保持身份一致性。

## 📋 模型概述

### 模型1：特征转换ControlNet (FeatureControlNet)

**功能**：将姿势信息和特征信息转换为目标角度的特征，保持身份一致性

**输入**：
- `features`: 源特征 [batch, 512]
- `pose`: 源姿势 [batch, 3] (欧拉角)
- `control_angle`: 控制角度（目标角度）[batch, 3] (欧拉角)

**输出**：
- `output_features`: 目标角度的特征 [batch, 512]（保持身份一致性）

**架构**：
```
输入特征 + 姿势
    ↓
主网络（特征转换）
    ↓
控制分支（接收控制角度）
    ↓
零卷积（注入控制信号）
    ↓
身份保护层
    ↓
输出特征
```

### 模型2：图像生成ControlNet (ImageControlNet)

**功能**：从图片生成目标角度的图片，受姿势控制

**输入**：
- `image`: 源图像 [batch, 3, H, W]
- `target_pose`: 目标姿势（控制姿势）[batch, 3] (欧拉角)

**输出**：
- `output_image`: 目标角度的图像 [batch, 3, H, W]

**架构**：
```
输入图像
    ↓
图像编码器（提取特征和姿势）
    ↓
控制分支（接收目标姿势）
    ↓
零卷积（注入控制信号）
    ↓
图像生成器
    ↓
输出图像
```

---

## 🏗️ 文件结构

```
controlnetlike/
├── __init__.py                    # 模块导出
├── models_feature_controlnet.py   # 模型1：特征转换ControlNet
├── models_image_controlnet.py    # 模型2：图像生成ControlNet
├── dataset_feature.py             # 模型1的数据集
├── dataset_image.py               # 模型2的数据集
├── train_feature_controlnet.py    # 模型1的训练脚本
├── train_image_controlnet.py     # 模型2的训练脚本
└── README.md                      # 本文档
```

---

## 🚀 快速开始

### 模型1：特征转换ControlNet

#### 训练

```bash
python train_transformer3D/controlnetlike/train_feature_controlnet.py \
    --data_dir train/datas/file \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --identity_loss_weight 0.3
```

#### 使用

```python
from train_transformer3D.controlnetlike import FeatureControlNet
import torch

# 创建模型
model = FeatureControlNet(
    feature_dim=512,
    pose_dim=3,
    hidden_dim=512,
    num_main_layers=3,
    num_control_layers=3,
    freeze_main=False  # 是否冻结主网络
)

# 前向传播
source_features = torch.randn(4, 512)  # 源特征
source_pose = torch.randn(4, 3)       # 源姿势
target_angle = torch.randn(4, 3)      # 目标角度（控制角度）

output_features, _ = model(
    features=source_features,
    pose=source_pose,
    control_angle=target_angle
)

print(f"输出特征: {output_features.shape}")  # [4, 512]
```

### 模型2：图像生成ControlNet

#### 训练

```bash
python train_transformer3D/controlnetlike/train_image_controlnet.py \
    --data_dir train/datas/file \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --image_size 112
```

#### 使用

```python
from train_transformer3D.controlnetlike import ImageControlNet
import torch

# 创建模型
model = ImageControlNet(
    feature_dim=512,
    pose_dim=3,
    image_size=112,
    in_channels=3,
    num_control_layers=3,
    freeze_generator=False  # 是否冻结生成器
)

# 前向传播
source_image = torch.randn(4, 3, 112, 112)  # 源图像
target_pose = torch.randn(4, 3)             # 目标姿势（控制姿势）

output_image, _, source_pose = model(
    image=source_image,
    target_pose=target_pose,
    return_control_signal=False,
    return_source_pose=True
)

print(f"输出图像: {output_image.shape}")  # [4, 3, 112, 112]
print(f"提取的源姿势: {source_pose.shape}")  # [4, 3]
```

---

## 📊 核心特性

### 1. 零卷积（Zero Convolution）

ControlNet的核心创新：
- **初始时输出为零**：不干扰主网络
- **训练过程中逐渐学习**：控制信号逐渐生效
- **精确控制**：可以精确控制生成过程

### 2. 身份一致性保护

**模型1**：
- 身份保护层：确保输出特征保持身份信息
- 身份一致性损失：在训练中约束身份信息

**模型2**：
- 通过特征编码保持身份信息
- 图像生成过程中保持身份一致性

### 3. 可冻结主网络

- **模型1**：可以冻结主网络，只训练控制分支
- **模型2**：可以冻结图像生成器，只训练控制分支
- **优势**：保护预训练能力，只学习控制信号

---

## 🔧 训练参数

### 模型1训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | 必需 | 数据目录路径 |
| `--batch_size` | 32 | 批次大小 |
| `--epochs` | 100 | 训练轮数 |
| `--lr` | 1e-4 | 学习率 |
| `--identity_loss_weight` | 0.3 | 身份一致性损失权重 |
| `--freeze_main` | False | 是否冻结主网络 |
| `--loss_type` | mse | 损失函数类型（mse/cosine/combined） |

### 模型2训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | 必需 | 数据目录路径 |
| `--image_dir` | None | 图片目录（如果与data_dir不同） |
| `--batch_size` | 16 | 批次大小（图像生成需要更多内存） |
| `--epochs` | 100 | 训练轮数 |
| `--lr` | 1e-4 | 学习率 |
| `--image_size` | 112 | 图像尺寸 |
| `--pose_loss_weight` | 0.1 | 姿势预测损失权重 |
| `--freeze_generator` | False | 是否冻结图像生成器 |

---

## 📈 训练监控

### TensorBoard

启动TensorBoard查看训练进度：

```bash
# 模型1
tensorboard --logdir train_transformer3D/controlnetlike/logs_feature

# 模型2
tensorboard --logdir train_transformer3D/controlnetlike/logs_image
```

### 训练曲线

训练过程中会自动生成训练曲线图，包含：
- **模型1**：总损失、特征损失、身份损失、余弦相似度
- **模型2**：总损失、图像损失、姿势损失、PSNR

---

## 🎯 使用场景

### 模型1：特征转换ControlNet

**适用场景**：
- 需要精确控制特征转换的角度
- 需要保持身份一致性的特征生成
- 需要从不同角度生成特征

**示例**：
```python
# 给定侧面特征和姿势，生成正面特征
side_features = ...  # 侧面特征
side_pose = ...      # 侧面姿势
front_angle = [0, 0, 0]  # 正面角度

front_features = model(side_features, side_pose, front_angle)
```

### 模型2：图像生成ControlNet

**适用场景**：
- 需要从图片生成不同角度的图片
- 需要精确控制生成图片的姿势
- 需要保持身份一致性的图像生成

**示例**：
```python
# 给定侧面图片，生成正面图片
side_image = ...     # 侧面图片
front_pose = [0, 0, 0]  # 正面姿势

front_image = model(side_image, front_pose)
```

---

## ⚠️ 注意事项

### 1. 数据准备

**模型1**：
- 需要 `front_feature.npy`, `front_keypoints.npy`, `front_metadata.json`
- 需要 `video_feature.npy`, `video_keypoints.npy`, `video_metadata.json`

**模型2**：
- 需要元数据和关键点文件（同上）
- 需要原始图片路径（在metadata中）

### 2. 内存使用

- **模型1**：内存占用较小，可以使用较大的batch_size
- **模型2**：内存占用较大（图像生成），建议使用较小的batch_size（16或更小）

### 3. 训练策略

**推荐策略**：
1. **第一阶段**：冻结主网络/生成器，只训练控制分支（快速收敛）
2. **第二阶段**：解冻所有参数，端到端微调（精细优化）

---

## 📝 完整训练示例

### 模型1完整训练

```bash
python train_transformer3D/controlnetlike/train_feature_controlnet.py \
    --data_dir train/datas/file \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --identity_loss_weight 0.3 \
    --loss_type combined \
    --mse_weight 0.5 \
    --cosine_weight 0.5 \
    --use_amp \
    --save_dir train_transformer3D/controlnetlike/checkpoints_feature \
    --log_dir train_transformer3D/controlnetlike/logs_feature
```

### 模型2完整训练

```bash
python train_transformer3D/controlnetlike/train_image_controlnet.py \
    --data_dir train/datas/file \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --image_size 112 \
    --pose_loss_weight 0.1 \
    --use_amp \
    --save_dir train_transformer3D/controlnetlike/checkpoints_image \
    --log_dir train_transformer3D/controlnetlike/logs_image
```

---

## 🔍 技术细节

### 零卷积的工作原理

1. **初始化**：权重和偏置都初始化为零
2. **初始阶段**：控制信号输出为零，不干扰主网络
3. **训练过程**：控制信号逐渐学习，开始影响主网络
4. **最终效果**：精确控制生成过程，同时保持主网络能力

### 身份一致性保护

**模型1**：
- 身份保护层：`0.7 * fused_features + 0.3 * identity_features`
- 身份一致性损失：`(1 - cosine_similarity) * weight`

**模型2**：
- 通过特征编码保持身份信息
- 图像生成过程中保持身份一致性

---

生成时间: 2024-12-16
