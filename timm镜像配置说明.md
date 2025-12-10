# timm 库镜像配置说明

## 问题描述

`timm` 库在下载模型时可能仍然尝试连接 `huggingface.co` 而不是镜像，导致连接超时。

## 原因分析

`timm` 库使用 `huggingface_hub` 下载模型，而 `huggingface_hub` 需要：
1. **在导入 `timm` 之前**设置 `HF_ENDPOINT` 环境变量
2. 环境变量必须在 `huggingface_hub` 初始化之前设置

## 解决方案

### 方案 1: 在脚本开头设置（推荐）

代码已自动在文件顶部设置，确保在导入 `timm` 之前：

```python
import os

# 必须在导入 timm 之前设置！
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'

import timm  # 现在可以安全导入
```

### 方案 2: 使用环境变量（运行前设置）

```bash
# Linux/Mac
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=300
python train_transformer/train.py ...

# Windows (PowerShell)
$env:HF_ENDPOINT="https://hf-mirror.com"
$env:HF_HUB_ENABLE_HF_TRANSFER="0"
$env:HF_HUB_DOWNLOAD_TIMEOUT="300"
python train_transformer/train.py ...
```

### 方案 3: 使用 setup_mirrors 模块

```python
# 在脚本开头导入
import setup_mirrors  # 会自动设置所有镜像环境变量

# 然后正常导入其他库
import timm
```

## 已更新的文件

以下文件已自动在导入 `timm` 之前设置镜像：

- ✅ `feature_extractor.py`
- ✅ `train/model.py`
- ✅ `test_file2.py`
- ✅ `add_similarity_to_landmarks.py`
- ✅ `train_transformer/dataset.py`
- ✅ `train_transformer/train.py`

## 验证配置

### 检查环境变量

```python
import os
print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '未设置')}")
```

### 测试 timm 下载

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import timm

# 尝试加载模型
model = timm.create_model('vit_base_patch16_224', pretrained=True)
print("✓ 模型下载成功")
```

## 常见错误

### 错误 1: Connection to huggingface.co timed out

**原因**: `HF_ENDPOINT` 未设置或设置太晚

**解决**: 确保在导入 `timm` 之前设置环境变量

### 错误 2: Max retries exceeded

**原因**: 网络连接问题或镜像不可用

**解决**:
1. 检查镜像是否可用
2. 增加超时时间：`export HF_HUB_DOWNLOAD_TIMEOUT=600`
3. 使用代理

### 错误 3: LocalEntryNotFoundError

**原因**: 模型不在本地缓存，且无法从镜像下载

**解决**:
1. 检查 `HF_ENDPOINT` 是否正确设置
2. 手动下载模型到缓存目录
3. 使用代理

## 手动下载模型

如果自动下载失败，可以手动下载：

```bash
# 1. 找到模型 URL（从错误信息中）
# 例如: https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k/resolve/main/pytorch_model.bin

# 2. 使用镜像 URL 下载
# 将 huggingface.co 替换为 hf-mirror.com
wget https://hf-mirror.com/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k/resolve/main/pytorch_model.bin

# 3. 放置到缓存目录
# ~/.cache/huggingface/hub/models--timm--vit_base_patch16_224.augreg2_in21k_ft_in1k/
```

## 推荐配置

在运行训练脚本前，设置以下环境变量：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_DOWNLOAD_RETRIES=5
```

或使用 `setup_mirrors.py`:

```python
import setup_mirrors  # 自动设置所有镜像
```

