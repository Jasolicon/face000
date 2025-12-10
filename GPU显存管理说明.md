# GPU显存管理说明

## 关于共享GPU显存

### 什么是共享GPU显存？

共享GPU显存（Shared GPU Memory）是Windows系统在GPU专用显存不足时，自动使用系统内存作为补充的机制。

### 当前代码的显存管理

**已添加的显存管理功能：**

1. **显存使用比例控制** (`--memory_fraction`)
   - 可以限制PyTorch使用的GPU显存比例
   - 例如：`--memory_fraction 0.8` 表示只使用80%的显存
   - 这可以避免显存溢出，减少使用共享显存的可能性

2. **TF32加速** (`--allow_tf32`)
   - 在支持的GPU上启用TensorFloat-32加速
   - 可以在不损失太多精度的情况下提升性能

### 共享显存的使用

**重要说明：**

1. **PyTorch无法直接控制共享显存**
   - 共享显存是由Windows系统自动管理的
   - 当专用显存不足时，系统会自动使用共享显存
   - 但共享显存速度很慢，不适合训练

2. **如何避免使用共享显存**
   - ✅ 减小batch_size：`--batch_size 8` 或更小
   - ✅ 限制显存使用：`--memory_fraction 0.7`（使用70%显存）
   - ✅ 使用混合精度训练（FP16）- 需要代码支持
   - ✅ 减小图像尺寸：`--image_size 160`
   - ✅ 使用梯度累积来模拟更大的batch_size

3. **检查是否使用了共享显存**
   - 在Windows任务管理器中查看GPU使用情况
   - 如果看到"共享GPU内存"被使用，说明显存不足
   - 此时训练速度会显著下降

### 使用示例

#### 限制显存使用（推荐）

```bash
# 训练脚本 - 限制使用80%显存
python train/train.py \
    --data_dir train/datas \
    --batch_size 16 \
    --memory_fraction 0.8 \
    --allow_tf32

# Transformer训练 - 限制使用70%显存
python train_transformer/train.py \
    --features_224_dir features_224 \
    --batch_size 32 \
    --memory_fraction 0.7 \
    --allow_tf32
```

#### 显存不足时的解决方案

```bash
# 方案1：减小batch_size
python train/train.py --data_dir train/datas --batch_size 8

# 方案2：限制显存使用 + 减小batch_size
python train/train.py \
    --data_dir train/datas \
    --batch_size 8 \
    --memory_fraction 0.6

# 方案3：减小图像尺寸
python train/train.py \
    --data_dir train/datas \
    --batch_size 16 \
    --image_size 160
```

### 最佳实践

1. **监控显存使用**
   - 使用 `nvidia-smi`（Linux）或任务管理器（Windows）监控显存
   - 确保专用显存使用率不超过90%

2. **预留显存空间**
   - 建议使用 `--memory_fraction 0.8` 或更小
   - 为系统和其他进程预留显存空间

3. **避免共享显存**
   - 如果发现使用了共享显存，立即减小batch_size或memory_fraction
   - 共享显存会导致训练速度下降10-100倍

### 技术细节

- **专用显存（Dedicated Memory）**：GPU自带的快速显存
- **共享显存（Shared Memory）**：系统内存，速度较慢
- **PyTorch显存管理**：通过 `torch.cuda.set_per_process_memory_fraction()` 控制
- **系统级共享显存**：由Windows系统自动管理，PyTorch无法直接控制

### 总结

✅ **可以使用共享显存**：系统会自动使用，但性能很差

❌ **不建议依赖共享显存**：应该通过调整参数避免使用

✅ **推荐做法**：使用 `--memory_fraction` 限制显存使用，配合较小的batch_size


