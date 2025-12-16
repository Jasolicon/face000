# Universal网络数据读取说明

## 📋 数据读取匹配性

✅ **数据读取完全匹配**：`UniversalFaceTransformer` 使用的数据格式与现有 `Aligned3DFaceDataset` 完全兼容。

## 🔍 数据读取策略

### 1. 数据集结构

`Aligned3DFaceDataset` 从以下文件读取数据：

```
train/datas/file/
├── front_feature.npy          # 正面图特征 [N, 512]
├── front_keypoints.npy        # 正面图关键点 [N, 5, 3]（已归一化）
├── front_metadata.json        # 正面图元数据（包含原始关键点、姿态等）
├── video_feature.npy          # 视频帧特征 [M, 512]
├── video_keypoints.npy        # 视频帧关键点 [M, 5, 3]（已归一化）
└── video_metadata.json        # 视频帧元数据（包含原始关键点、姿态等）
```

### 2. 样本构建策略

**核心思想**：通过 `person_name` 建立正面图和视频帧的配对关系。

```python
# 每个正面图对应所有该人的视频帧
for front_idx in front_indices:
    for video_idx in video_indices:
        samples.append({
            'front_idx': front_idx,
            'video_idx': video_idx,
            'person_name': person_name
        })
```

**示例**：
- 如果有 36 个正面图（36个人）
- 每个平均有 282 个视频帧
- 总共会构建约 36 × 282 = 10,152 个训练样本

### 3. 数据加载模式

#### 模式1：内存加载（推荐，默认）

```python
load_in_memory=True
```

**特点**：
- 所有数据一次性加载到内存
- 访问速度快
- 适合数据集不太大的情况（<10GB）

**使用场景**：
- 数据集较小
- 需要快速训练
- 内存充足

#### 模式2：内存映射（大数据集）

```python
load_in_memory=False
```

**特点**：
- 使用 `mmap_mode='r'` 内存映射
- 按需加载，不占用全部内存
- 访问速度稍慢

**使用场景**：
- 数据集很大（>10GB）
- 内存有限
- 可以接受稍慢的加载速度

### 4. 数据返回格式

`__getitem__` 返回的字典包含：

```python
{
    'src': torch.Tensor,           # [512] 视频帧特征（侧面特征，输入）
    'tgt': torch.Tensor,           # [512] 正面图特征（正面特征，目标）
    'keypoints_3d': torch.Tensor,  # [5, 3] 视频帧关键点（Universal网络不使用）
    'pose': torch.Tensor,          # [3] 视频帧姿态 (yaw, pitch, roll)
    'angles': torch.Tensor,        # [3] 角度（兼容性，等于pose）
    'person_name': str,            # 人员名称（用于对比学习）
    'front_keypoints_3d': ...,     # Universal网络不使用
    'front_pose': ...,             # Universal网络不使用
    'front_angles': ...            # Universal网络不使用
}
```

### 5. Universal网络使用的字段

Universal网络训练时只使用以下字段：

| 字段 | 用途 | 形状 |
|------|------|------|
| `src` | 模型输入（侧面特征） | [batch, 512] |
| `tgt` | 目标特征（正面特征） | [batch, 512] |
| `pose` | 姿态角度（用于姿态感知） | [batch, 3] |
| `person_name` | 人员ID（用于对比学习） | [batch] |

**注意**：`keypoints_3d` 字段虽然会被返回，但Universal网络不使用它（因为只使用姿态作为位置编码）。

### 6. 数据配对策略

#### 一对一配对

每个训练样本包含：
- **输入**：一个视频帧的特征（侧面）
- **目标**：对应人员的正面图特征（正面）

**配对规则**：
- 同一人的所有视频帧都配对到该人的正面图
- 如果一个人有 1 个正面图和 282 个视频帧，会产生 282 个训练样本

#### 数据增强（可选）

当前实现中没有数据增强，但可以添加：

```python
# 在 __getitem__ 中可以添加：
# - 特征噪声（轻微）
# - 姿态角度扰动（用于数据增强）
```

### 7. 角度过滤策略

支持按yaw角度过滤样本：

```python
# 只保留大角度样本
min_yaw_angle=15  # 只保留 |yaw| >= 15° 的样本

# 只保留特定角度范围
min_yaw_angle=15
max_yaw_angle=60  # 只保留 15° <= |yaw| <= 60° 的样本
```

**过滤逻辑**：
```python
abs_yaw = abs(yaw_angle)
if abs_yaw < min_yaw_angle:  # 排除
    continue
if abs_yaw > max_yaw_angle:  # 排除
    continue
```

### 8. 数据集分割策略

使用 `create_train_val_test_dataloaders` 按 `person_name` 分割：

```python
train_ratio=0.6  # 60% 的人用于训练
val_ratio=0.3    # 30% 的人用于验证
test_ratio=0.1   # 10% 的人用于测试
```

**分割方式**：
- 按人员分割（不是按样本分割）
- 确保同一人的所有样本都在同一个集合中
- 避免数据泄露

**示例**：
- 36 个人
- 训练集：21 人（60%）
- 验证集：10 人（30%）
- 测试集：5 人（10%）

### 9. 关键点归一化策略

**当前策略**：使用图片自己的中心点归一化

```python
# 对每个图片：
image_center = mean(landmarks_3d)  # 5个关键点的平均值
relative_landmarks = landmarks_3d - image_center
normalized = relative_landmarks * standard_scale
```

**优点**：
- 去除位置偏差
- 只保留形状信息
- 每个图片独立归一化

**注意**：Universal网络不使用关键点，所以这个归一化策略对Universal网络没有影响。

### 10. 数据加载流程

```
1. 初始化数据集
   ├── 加载元数据（JSON）
   ├── 加载特征和关键点（NPY）
   ├── 构建样本索引映射（按person_name）
   └── 应用角度过滤（如果指定）

2. 获取样本（__getitem__）
   ├── 根据索引获取 front_idx 和 video_idx
   ├── 加载特征：src（视频帧），tgt（正面图）
   ├── 加载姿态：pose（视频帧姿态）
   ├── 加载关键点（Universal网络不使用）
   └── 返回字典

3. DataLoader批处理
   ├── 自动批处理（batch_size）
   ├── 自动转换为torch.Tensor
   └── 返回批次字典
```

## ✅ 数据匹配验证

### 训练脚本期望的格式

```python
batch = {
    'src': [batch_size, 512],      # ✅ 匹配
    'tgt': [batch_size, 512],      # ✅ 匹配
    'pose': [batch_size, 3],       # ✅ 匹配
    'person_name': [batch_size]     # ✅ 匹配
}
```

### 模型输入格式

```python
# UniversalFaceTransformer.forward()
features: [batch, 512]     # ✅ 来自 batch['src']
pose_angles: [batch, 3]    # ✅ 来自 batch['pose']
```

### 损失函数期望的格式

```python
targets = {
    'tgt_features': [batch, 512],    # ✅ 来自 batch['tgt']
    'pose_labels': [batch, 3],        # ✅ 来自 batch['pose']
    'person_names': [batch]            # ✅ 来自 batch['person_name']
}
```

## 📊 数据统计示例

根据你的日志，典型的数据分布：

```
总人数: 36
训练集: 21 人, 5790 个样本 (57.0%)
验证集: 10 人, 3163 个样本 (31.2%)
测试集: 5 人, 1198 个样本 (11.8%)

特征维度: 512 (InsightFace)
姿态维度: 3 (yaw, pitch, roll)
```

## 🔧 数据读取优化建议

### 1. 内存优化

如果内存不足，可以：
- 使用 `load_in_memory=False`
- 减少 `batch_size`
- 减少 `num_workers`

### 2. 速度优化

如果加载速度慢，可以：
- 使用 `load_in_memory=True`
- 增加 `num_workers`
- 使用 `pin_memory=True`（GPU训练）

### 3. 数据质量

确保数据质量：
- 检查元数据中的 `landmarks_3d_original` 是否存在
- 验证特征维度一致性
- 检查姿态角度范围是否合理

## 💡 总结

1. **数据格式完全匹配**：Universal网络可以直接使用现有的数据集
2. **配对策略清晰**：通过 `person_name` 建立配对关系
3. **加载方式灵活**：支持内存加载和内存映射
4. **角度过滤支持**：可以只使用大角度数据训练
5. **分割策略合理**：按人员分割，避免数据泄露

**无需修改数据集代码**，可以直接使用！
