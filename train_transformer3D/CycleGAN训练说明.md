# CycleGAN训练说明

## 📐 架构概述

CycleGAN架构将现有的Transformer模型作为生成器，添加判别器进行对抗训练，提高生成质量。

### 核心组件

1. **生成器 G_AB**：侧面特征 → 正面特征（使用现有Transformer模型）
2. **生成器 G_BA**：正面特征 → 侧面特征（反向生成器）
3. **判别器 D_A**：区分真实和生成的侧面特征
4. **判别器 D_B**：区分真实和生成的正面特征

### 损失函数

1. **对抗损失**：生成器希望欺骗判别器，判别器希望正确分类
2. **循环一致性损失**：侧面→正面→侧面应该恢复原样
3. **身份损失**（可选）：正面→正面应该保持不变

## 🚀 使用方法

### 基本训练命令

```bash
python train_transformer3D/gan_train.py \
    --data_dir train/datas/file \
    --batch_size 16 \
    --epochs 100 \
    --generator_type decoder_only \
    --discriminator_type patch \
    --lr_G 2e-4 \
    --lr_D 2e-4 \
    --lambda_cycle 10.0 \
    --lambda_identity 0.5 \
    --use_lsgan \
    --use_mixed_precision
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--generator_type` | `decoder_only` | 生成器类型（decoder_only/encoder_decoder/angle_warping） |
| `--discriminator_type` | `patch` | 判别器类型（simple/patch） |
| `--batch_size` | 16 | 批次大小（GAN训练建议较小） |
| `--lr_G` | 2e-4 | 生成器学习率 |
| `--lr_D` | 2e-4 | 判别器学习率 |
| `--lambda_cycle` | 10.0 | 循环一致性损失权重 |
| `--lambda_identity` | 0.5 | 身份损失权重（0表示禁用） |
| `--lambda_adversarial` | 1.0 | 对抗损失权重 |
| `--use_lsgan` | True | 使用LSGAN损失（更稳定） |

### 不同生成器类型

#### 1. Transformer Decoder-Only（推荐）

```bash
python train_transformer3D/gan_train.py \
    --generator_type decoder_only \
    --data_dir train/datas/file \
    --batch_size 16 \
    --epochs 100 \
    --lr_G 2e-4 \
    --lr_D 2e-4
```

#### 2. Transformer Encoder-Decoder

```bash
python train_transformer3D/gan_train.py \
    --generator_type encoder_decoder \
    --data_dir train/datas/file \
    --batch_size 16 \
    --epochs 100 \
    --lr_G 2e-4 \
    --lr_D 2e-4
```

#### 3. 角度条件仿射变换

```bash
python train_transformer3D/gan_train.py \
    --generator_type angle_warping \
    --data_dir train/datas/file \
    --batch_size 16 \
    --epochs 100 \
    --lr_G 2e-4 \
    --lr_D 2e-4
```

## 📊 训练策略

### 1. 学习率设置

- **生成器学习率**：`2e-4`（与判别器相同或稍小）
- **判别器学习率**：`2e-4`
- **Adam参数**：`beta1=0.5, beta2=0.999`（GAN标准配置）

### 2. 损失权重调整

#### 标准配置（推荐）

```bash
--lambda_cycle 10.0 \
--lambda_identity 0.5 \
--lambda_adversarial 1.0
```

#### 强调循环一致性

```bash
--lambda_cycle 20.0 \
--lambda_identity 0.5 \
--lambda_adversarial 1.0
```

#### 禁用身份损失

```bash
--lambda_cycle 10.0 \
--lambda_identity 0.0 \
--lambda_adversarial 1.0
```

### 3. 批次大小

- **小批次（16）**：更稳定，但训练慢
- **中等批次（32）**：平衡速度和稳定性
- **大批次（64）**：可能不稳定，不推荐

### 4. 判别器类型

#### Simple Discriminator（简单）

- 参数量少
- 训练快
- 判别能力较弱

#### Patch Discriminator（推荐）

- 参数量多
- 判别能力更强
- 需要 `d_model` 能被 `patch_size` 整除

## 🔬 技术细节

### 对抗损失

#### LSGAN（推荐）

```python
# 生成器损失：希望判别器输出接近1
loss_G = mean((D(fake) - 1)^2)

# 判别器损失：真实接近1，生成接近0
loss_D = mean((D(real) - 1)^2) + mean(D(fake)^2)
```

#### BCE损失

```python
# 使用BCEWithLogitsLoss
loss_G = BCE(D(fake), 1)
loss_D = BCE(D(real), 1) + BCE(D(fake), 0)
```

### 循环一致性损失

```python
# 侧面→正面→侧面
rec_side = G_BA(G_AB(side))
loss_cycle_A = L1(rec_side, side)

# 正面→侧面→正面
rec_front = G_AB(G_BA(front))
loss_cycle_B = L1(rec_front, front)

loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
```

### 身份损失

```python
# 正面→正面应该保持不变
id_front = G_AB(front)
loss_id = L1(id_front, front)
```

## 📈 训练监控

### TensorBoard

```bash
tensorboard --logdir train_transformer3D/gan_logs
```

### 关键指标

1. **Loss_G**：生成器总损失（应该下降）
2. **Loss_D_A / Loss_D_B**：判别器损失（应该稳定）
3. **Loss_Cycle**：循环一致性损失（应该下降）
4. **Loss_Identity**：身份损失（应该很小）

### 健康训练信号

- ✅ 生成器损失和判别器损失都在下降
- ✅ 循环一致性损失持续下降
- ✅ 判别器不能完全区分真实和生成（准确率约50-70%）

### 异常信号

- ⚠️ 判别器损失为0：判别器太强，生成器无法学习
- ⚠️ 生成器损失不下降：学习率太小或模型容量不足
- ⚠️ 循环一致性损失不下降：循环一致性权重太小

## 🎯 训练技巧

### 1. 渐进式训练

```python
# 前50个epoch：只训练生成器（lambda_adversarial=0）
# 后50个epoch：正常训练
```

### 2. 学习率衰减

```python
# 每20个epoch衰减学习率
scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=20, gamma=0.5)
scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=20, gamma=0.5)
```

### 3. 判别器更新频率

```python
# 每2个batch更新一次判别器（更稳定）
if batch_idx % 2 == 0:
    update_discriminator()
```

### 4. 梯度惩罚（可选）

```python
# WGAN-GP风格，提高训练稳定性
def gradient_penalty(discriminator, real, fake):
    # ... 实现梯度惩罚
```

## ⚠️ 注意事项

1. **批次大小**：GAN训练建议使用较小的批次（16-32）
2. **学习率**：生成器和判别器学习率应该相同或接近
3. **损失平衡**：确保各损失项在同一数量级
4. **梯度裁剪**：生成器梯度裁剪到1.0，防止梯度爆炸
5. **混合精度**：可以使用，但要注意数值稳定性

## 📝 完整训练示例

```bash
python train_transformer3D/gan_train.py \
    --data_dir train/datas/file \
    --batch_size 16 \
    --num_workers 4 \
    --epochs 100 \
    --generator_type decoder_only \
    --discriminator_type patch \
    --d_model 512 \
    --lr_G 2e-4 \
    --lr_D 2e-4 \
    --beta1 0.5 \
    --beta2 0.999 \
    --lambda_cycle 10.0 \
    --lambda_identity 0.5 \
    --lambda_adversarial 1.0 \
    --use_lsgan \
    --use_mixed_precision \
    --save_dir train_transformer3D/gan_checkpoints \
    --log_dir train_transformer3D/gan_logs
```

## 🔗 相关文件

- `cyclegan.py`: CycleGAN架构定义
- `gan_train.py`: GAN训练脚本
- `models_3d.py`: Transformer生成器模型
- `models_angle_warping.py`: 角度条件仿射变换生成器

## 📚 参考文献

- **CycleGAN**: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
- **LSGAN**: Least Squares Generative Adversarial Networks
- **PatchGAN**: Image-to-Image Translation with Conditional Adversarial Networks
