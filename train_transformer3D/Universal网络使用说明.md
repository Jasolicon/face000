# 通用人脸姿态不变网络使用说明

## 📋 网络架构概述

`UniversalFaceTransformer` 是一个融合了特征解耦、对比学习和姿态不变性思想的人脸正面化网络。

### 核心创新点

1. **特征解耦架构**
   - 正交投影层强制身份和姿态特征分离
   - 姿态感知的Transformer编码器
   - 身份增强模块去除姿态影响

2. **姿态感知机制**
   - `PoseAwareAttention`: 姿态引导的注意力
   - `PoseNormalizationLayer`: 姿态自适应的归一化
   - 姿态原型记忆库

3. **多任务学习**
   - 身份相似度损失（正面化目标）
   - 姿态估计损失（辅助监督）
   - 对比学习损失（同一人不同姿态）
   - 正交约束损失（特征解耦）
   - 重建损失（特征重建）

## 🚀 快速开始

### 基础训练命令

```bash
C:/Users/62487/.conda/envs/llm/python.exe train_transformer3D/train_universal.py --data_dir train/datas/file --batch_size 32 --epochs 150 --lr 1e-4
```

### 完整推荐命令

```bash
C:/Users/62487/.conda/envs/llm/python.exe train_transformer3D/train_universal.py --data_dir train/datas/file --batch_size 32 --epochs 150 --lr 1e-4 --feat_dim 512 --id_dim 256 --pose_dim 128 --transformer_depth 4 --transformer_heads 8 --lambda_id 1.0 --lambda_pose 0.5 --lambda_ortho 0.1 --lambda_contrast 0.3 --lambda_reconstruction 0.2 --use_mixed_precision --min_yaw_angle 15
```

## 📊 参数说明

### 模型参数

- `--feat_dim`: 特征维度（默认512，InsightFace特征维度）
- `--id_dim`: 身份特征维度（默认256）
- `--pose_dim`: 姿态特征维度（默认128）
- `--num_pose_bins`: 姿态原型数量（默认36）
- `--transformer_depth`: Transformer深度（默认4）
- `--transformer_heads`: 注意力头数（默认8）
- `--transformer_mlp_dim`: Transformer MLP维度（默认1024）
- `--dropout`: Dropout比率（默认0.1）

### 损失权重

- `--lambda_id`: 身份相似度损失权重（默认1.0）
- `--lambda_pose`: 姿态估计损失权重（默认0.5）
- `--lambda_ortho`: 正交约束损失权重（默认0.1）
- `--lambda_contrast`: 对比学习损失权重（默认0.3）
- `--lambda_reconstruction`: 重建损失权重（默认0.2）

### 训练参数

- `--batch_size`: 批次大小（默认32）
- `--epochs`: 训练轮数（默认150）
- `--lr`: 学习率（默认1e-4）
- `--weight_decay`: 权重衰减（默认1e-5）
- `--use_mixed_precision`: 使用混合精度训练
- `--min_yaw_angle`: 最小yaw角度阈值（度）
- `--max_yaw_angle`: 最大yaw角度阈值（度）

## 📈 训练监控

训练过程中会记录以下指标到TensorBoard：

- `Train/Loss`: 训练总损失
- `Train/Loss_id_similarity`: 身份相似度损失
- `Train/Loss_pose`: 姿态估计损失
- `Train/Loss_ortho`: 正交约束损失
- `Train/Loss_contrast`: 对比学习损失
- `Train/Loss_reconstruction`: 重建损失
- `Val/Loss`: 验证总损失
- `Val/CosineSimilarity`: 验证余弦相似度

## 🔍 模型输出

### 训练模式

```python
outputs = model(features, pose_angles, mode='train')
# 返回:
# {
#     'id_features': [batch, id_dim],      # 身份特征
#     'pose_features': [batch, pose_dim],  # 姿态特征
#     'pose_angles': [batch, 3],          # 估计的姿态角度
#     'base_features': [batch, feat_dim],  # 基础特征
#     ...
# }
```

### 推理模式

```python
outputs = model(features, pose_angles=None, mode='inference')
# 返回:
# {
#     'id_features': [batch, id_dim],      # 归一化的身份特征（用于识别）
#     'pose_angles': [batch, 3],          # 估计的姿态角度
#     'pose_features': [batch, pose_dim] # 归一化的姿态特征
# }
```

## 💡 使用建议

### 1. 损失权重调整

根据训练情况调整损失权重：

- **如果身份特征质量不好**：增加 `--lambda_id`
- **如果姿态估计不准确**：增加 `--lambda_pose`
- **如果特征解耦不充分**：增加 `--lambda_ortho`
- **如果对比学习效果差**：增加 `--lambda_contrast`

### 2. 模型容量调整

- **小数据集**：减少 `--transformer_depth` 和 `--transformer_mlp_dim`
- **大数据集**：增加 `--id_dim` 和 `--pose_dim`
- **显存不足**：减少 `--batch_size` 和 `--transformer_mlp_dim`

### 3. 角度过滤

只使用大角度数据训练（推荐）：

```bash
--min_yaw_angle 15
```

## 📝 示例命令

### 基础训练

```bash
C:/Users/62487/.conda/envs/llm/python.exe train_transformer3D/train_universal.py --data_dir train/datas/file --batch_size 32 --epochs 150 --lr 1e-4
```

### 大角度数据训练

```bash
C:/Users/62487/.conda/envs/llm/python.exe train_transformer3D/train_universal.py --data_dir train/datas/file --batch_size 32 --epochs 150 --lr 1e-4 --min_yaw_angle 15 --use_mixed_precision
```

### 高容量模型训练

```bash
C:/Users/62487/.conda/envs/llm/python.exe train_transformer3D/train_universal.py --data_dir train/datas/file --batch_size 16 --epochs 200 --lr 8e-5 --id_dim 512 --pose_dim 256 --transformer_depth 6 --transformer_mlp_dim 2048 --use_mixed_precision
```

### 恢复训练

```bash
C:/Users/62487/.conda/envs/llm/python.exe train_transformer3D/train_universal.py --data_dir train/datas/file --resume train_transformer3D/checkpoints_universal/best_model.pth
```

## 🔬 模型验证

训练完成后，可以使用诊断工具验证模型效果：

```bash
C:/Users/62487/.conda/envs/llm/python.exe diagnose_model_effectiveness.py --model_path train_transformer3D/checkpoints_universal/best_model.pth --data_dir train/datas/file --model_type universal
```

（✓ 已支持：`diagnose_model_effectiveness.py` 已更新以支持 `universal` 模型类型）

## 📊 预期效果

使用这个网络架构，预期可以达到：

1. **更好的特征解耦**：身份特征和姿态特征更独立
2. **更强的姿态不变性**：不同姿态下的身份特征更一致
3. **更好的正面化效果**：生成的正面特征质量更高
4. **更稳定的训练**：多任务学习使训练更稳定

## ⚠️ 注意事项

1. **内存占用**：Transformer架构比之前的模型占用更多内存
2. **训练时间**：由于多任务学习，训练时间可能稍长
3. **超参数敏感**：损失权重需要仔细调整
4. **数据要求**：需要配对数据（侧面特征和对应的正面特征）

## 🔄 与之前模型的对比

| 特性 | TransformerDecoderOnly3D | UniversalFaceTransformer |
|------|-------------------------|-------------------------|
| **特征解耦** | ❌ | ✅ |
| **对比学习** | ❌ | ✅ |
| **姿态感知注意力** | ✅ (简单) | ✅ (高级) |
| **多任务学习** | ❌ | ✅ |
| **模型复杂度** | 中等 | 较高 |
| **训练稳定性** | 中等 | 较高 |
