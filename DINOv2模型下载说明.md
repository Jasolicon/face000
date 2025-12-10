# DINOv2 模型下载说明

## 功能说明

`download_dinov2.py` 脚本用于下载 DINOv2 模型，支持：
- 下载小模型（dinov2_vits14，384维）
- 下载768维模型（dinov2_vitb14）
- 下载到指定位置

## 使用方法

### 1. 下载小模型和768维模型到指定位置（推荐）

```bash
# 下载到当前目录下的 models 文件夹
python download_dinov2.py --save_dir ./models

# 下载到指定路径
python download_dinov2.py --save_dir D:/models/dinov2

# 或者使用 --all 参数（默认行为）
python download_dinov2.py --all --save_dir ./models
```

### 2. 只下载特定模型

```bash
# 只下载小模型（384维）
python download_dinov2.py --model dinov2_vits14 --save_dir ./models

# 只下载768维模型
python download_dinov2.py --model dinov2_vitb14 --save_dir ./models
```

### 3. 只下载到默认缓存位置（不保存到指定位置）

```bash
# 下载小模型和768维模型到默认缓存
python download_dinov2.py

# 或指定模型
python download_dinov2.py --model dinov2_vitb14
```

## 模型信息

| 模型名称 | 特征维度 | 参数量 | 大小 | 说明 |
|---------|---------|--------|------|------|
| dinov2_vits14 | 384 | ~22M | ~88MB | 小模型，速度快 |
| dinov2_vitb14 | 768 | ~86M | ~344MB | 中等模型，**推荐用于训练** |
| dinov2_vitl14 | 1024 | ~300M | ~1.2GB | 大模型 |
| dinov2_vitg14 | 1536 | ~1.1B | ~4.4GB | 超大模型 |

## 下载位置说明

### 默认缓存位置

模型会自动下载到 PyTorch 的默认缓存目录：
- **Windows**: `C:\Users\<用户名>\.cache\torch\hub\checkpoints\`
- **Linux/Mac**: `~/.cache/torch/hub/checkpoints/`

### 指定保存位置

如果使用 `--save_dir` 参数，模型会：
1. 下载到默认缓存位置（PyTorch 自动管理）
2. **同时**保存到指定目录（包括 state_dict 格式和原始文件）

## 使用示例

### 示例 1：下载到项目目录

```bash
# 在项目根目录下创建 models 文件夹并下载
python download_dinov2.py --save_dir ./models/dinov2
```

下载后的目录结构：
```
models/
└── dinov2/
    ├── dinov2_vits14.pth          # 小模型 state_dict
    ├── dinov2_vitb14.pth          # 768维模型 state_dict
    └── [其他原始模型文件]
```

### 示例 2：下载到特定路径

```bash
# Windows
python download_dinov2.py --save_dir D:\Code\face000\models\dinov2

# Linux/Mac
python download_dinov2.py --save_dir /home/user/models/dinov2
```

### 示例 3：在代码中使用下载的模型

下载后，模型可以在代码中正常使用：

```python
from feature_extractor import DINOv2FeatureExtractor

# 使用小模型（384维）
extractor_small = DINOv2FeatureExtractor(model_name='dinov2_vits14')

# 使用768维模型（推荐用于训练）
extractor_768 = DINOv2FeatureExtractor(model_name='dinov2_vitb14')
```

PyTorch 会自动从缓存加载模型，无需手动指定路径。

## 注意事项

1. **网络连接**：首次下载需要网络连接，模型文件较大，请确保网络稳定
2. **磁盘空间**：确保有足够的磁盘空间（至少 500MB）
3. **下载时间**：根据网络速度，下载可能需要几分钟到十几分钟
4. **重复下载**：如果模型已存在于缓存中，不会重新下载

## 常见问题

### Q1: 下载失败怎么办？

**解决方案**：
1. 检查网络连接
2. 确保已安装 torch 和 torchvision
3. 重新运行脚本（支持断点续传）

### Q2: 如何验证模型下载成功？

运行脚本后，如果看到以下信息表示成功：
```
✓ 模型下载成功！
✓ 模型测试成功！
✓ 模型已保存到: [路径]
```

### Q3: 下载的模型文件在哪里？

- **默认位置**：`~/.cache/torch/hub/checkpoints/`（Windows: `C:\Users\<用户名>\.cache\torch\hub\checkpoints\`）
- **指定位置**：使用 `--save_dir` 参数指定的目录

### Q4: 如何从指定位置加载模型？

PyTorch 的 `torch.hub.load()` 会自动从缓存加载，无需手动指定路径。如果要从保存的 state_dict 加载：

```python
import torch
from torch.hub import load_state_dict_from_url

# 从保存的位置加载
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=False)
state_dict = torch.load('./models/dinov2/dinov2_vitb14.pth')
model.load_state_dict(state_dict)
```

## 推荐配置

对于训练任务，推荐：
- **使用 768维模型**（dinov2_vitb14）
- **下载到项目目录**：`python download_dinov2.py --save_dir ./models/dinov2`

这样可以：
1. 确保模型文件在项目中，便于版本控制（如果文件不太大）
2. 方便在不同环境间共享模型
3. 避免依赖系统缓存目录


