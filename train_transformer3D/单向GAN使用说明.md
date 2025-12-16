# 单向GAN使用说明

## 🎯 什么是单向GAN？

单向GAN是CycleGAN的简化版本，专注于**单向转换**任务：
- **输入**：斜向脸部特征 + 角度信息
- **输出**：正向脸部特征

### 架构对比

| 组件 | CycleGAN | 单向GAN |
|------|----------|---------|
| **生成器** | G_AB（侧面→正面）+ G_BA（正面→侧面） | 仅 G_AB（侧面→正面） |
| **判别器** | D_A（判断侧面）+ D_B（判断正面） | 仅 D_B（判断正面） |
| **损失** | 对抗损失 + 循环一致性损失 | 对抗损失 + 配对损失 |

## ✅ 优势

1. **更简单**：减少50%的模型参数（移除G_BA和D_A）
2. **训练更快**：每个batch只训练1个生成器和1个判别器
3. **更专注**：直接优化核心目标（侧面→正面）
4. **更稳定**：减少训练复杂度，降低训练难度

## 📝 使用方法

### 命令行参数

添加 `--unidirectional` 标志即可启用单向GAN：

```bash
python train_transformer3D/gan_train.py \
    --data_dir train/datas/file \
    --batch_size 32 \
    --epochs 100 \
    --lr_G 4e-4 \
    --lr_D 1e-4 \
    --lambda_cycle 10.0 \
    --lambda_identity 0.5 \
    --lambda_adversarial 2.0 \
    --unidirectional \
    --use_lsgan \
    --use_mixed_precision
```

### 关键参数说明

- `--unidirectional`：启用单向GAN模式
- `--lambda_cycle`：在单向模式下，这是**配对损失**权重（不是循环一致性）
- `--lambda_identity`：身份损失权重（可选）
- `--lambda_adversarial`：对抗损失权重（建议2.0）

## 🔄 训练流程

### 单向GAN训练流程

```
每个batch:
1. 训练判别器D_B：
   - 真实正面特征 → D_B → 接近1
   - 生成正面特征 → D_B → 接近0

2. 训练生成器G_AB：
   - 侧面特征 → G_AB → 生成正面特征
   - 对抗损失：希望D_B认为生成的正面特征是真实的
   - 配对损失：生成的正面特征应该接近真实的正面特征
```

### 损失函数

```python
# 生成器总损失
loss_G = (
    lambda_adversarial * loss_adv_G +  # 对抗损失
    lambda_cycle * loss_cycle +        # 配对损失（单向模式下）
    lambda_identity * loss_identity    # 身份损失（可选）
)

# 对抗损失
loss_adv_G = adversarial_loss(D_B(fake_front), target_is_real=True)

# 配对损失（单向模式下）
loss_cycle = L1(fake_front, front_features)  # 或CombinedLoss
```

## 📊 与CycleGAN的对比

### 训练时间

- **CycleGAN**：每个batch训练2个生成器和2个判别器
- **单向GAN**：每个batch训练1个生成器和1个判别器
- **速度提升**：约2倍

### 模型参数

- **CycleGAN**：G_AB + G_BA + D_A + D_B
- **单向GAN**：G_AB + D_B
- **参数减少**：约50%

### 适用场景

| 场景 | 推荐方案 |
|------|---------|
| 只需要侧面→正面 | ✅ 单向GAN |
| 需要双向转换 | CycleGAN |
| 需要循环一致性约束 | CycleGAN |
| 训练资源有限 | ✅ 单向GAN |
| 快速原型验证 | ✅ 单向GAN |

## ⚠️ 注意事项

1. **配对数据**：单向GAN需要配对数据（侧面和正面特征来自同一人）
2. **配对损失**：在单向模式下，`lambda_cycle`控制的是配对损失，不是循环一致性损失
3. **模型保存**：单向GAN的检查点不包含`optimizer_D_A_state_dict`
4. **恢复训练**：从CycleGAN检查点恢复时，需要移除`--unidirectional`标志

## 🎯 推荐配置

### 单向GAN推荐配置

```bash
python train_transformer3D/gan_train.py \
    --data_dir train/datas/file \
    --batch_size 32 \
    --epochs 150 \
    --lr_G 4e-4 \
    --lr_D 1e-4 \
    --lambda_cycle 10.0 \
    --lambda_identity 0.5 \
    --lambda_adversarial 2.0 \
    --unidirectional \
    --use_lsgan \
    --use_mixed_precision \
    --generator_type decoder_only
```

### 关键参数建议

- `lr_G = 4e-4`：生成器学习率（比判别器高）
- `lr_D = 1e-4`：判别器学习率（比生成器低）
- `lambda_adversarial = 2.0`：对抗损失权重（如果判别器太强可以增加）
- `lambda_cycle = 10.0`：配对损失权重（单向模式下）

## 📈 预期效果

使用单向GAN后，你应该看到：

1. **训练速度提升**：每个epoch时间减少约50%
2. **内存占用减少**：模型参数减少约50%
3. **训练更稳定**：减少训练复杂度
4. **效果相当**：如果只需要侧面→正面，效果应该与CycleGAN相当

## 🔍 监控指标

在单向GAN模式下，监控以下指标：

- `Loss_G`：生成器总损失
- `Loss_D_B`：判别器B损失（核心判别器）
- `Loss_Cycle`：配对损失（单向模式下）
- `Acc_D_B`：判别器B准确率（应该保持在50-70%之间）

**注意**：不会显示`Loss_D_A`和`Acc_D_A`（因为D_A不存在）
