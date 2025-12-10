# InsightFace 模型下载说明

## 问题描述

在下载 InsightFace 模型时可能遇到网络错误：
- `IncompleteRead`: 下载不完整
- `ChunkedEncodingError`: 连接中断
- `Connection broken`: 连接断开

## 解决方案

### 方案 1: 使用下载脚本（推荐）

运行专门的下载脚本，支持自动重试和断点续传：

```bash
# 下载模型（自动重试）
python download_insightface.py

# 检查模型是否已存在
python download_insightface.py --check

# 自定义重试次数和延迟
python download_insightface.py --max_retries 10 --retry_delay 10
```

**特点**：
- ✅ 自动重试（默认5次）
- ✅ 支持断点续传（已下载部分不会丢失）
- ✅ 指数退避策略（每次重试延迟翻倍）
- ✅ 详细的错误提示

### 方案 2: 多次运行训练脚本

如果下载中断，直接重新运行训练脚本即可：

```bash
# 第一次运行（下载中断）
python train_transformer/train.py ...

# 再次运行（会从断点继续下载）
python train_transformer/train.py ...
```

InsightFace 支持断点续传，已下载的部分不会丢失。

### 方案 3: 手动下载模型

如果网络非常不稳定，可以手动下载：

#### 步骤 1: 找到模型目录

```bash
# Linux/Mac
~/.insightface/models/buffalo_l/

# Windows
C:\Users\<用户名>\.insightface\models\buffalo_l\
```

#### 步骤 2: 下载模型文件

从以下地址下载模型文件：
- GitHub Releases: https://github.com/deepinsight/insightface/releases
- 或使用其他镜像源

需要的文件：
- `det_10g.onnx` - 人脸检测模型
- `w600k_r50.onnx` - 人脸识别模型
- `genderage.onnx` - 性别年龄识别模型（可选）

#### 步骤 3: 放置模型文件

将下载的文件放到模型目录中：
```
~/.insightface/models/buffalo_l/
├── det_10g.onnx
├── w600k_r50.onnx
└── genderage.onnx
```

### 方案 4: 使用环境变量指定模型路径

如果模型在其他位置，可以设置环境变量：

```bash
# Linux/Mac
export INSIGHTFACE_ROOT=/path/to/models

# Windows
set INSIGHTFACE_ROOT=D:\path\to\models
```

### 方案 5: 使用代理

如果网络受限，可以设置代理：

```bash
# 设置 HTTP 代理
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080

# 然后运行下载脚本
python download_insightface.py
```

## 模型信息

### buffalo_l（推荐）

- **大小**: ~288 MB
- **精度**: 高
- **速度**: 中等
- **用途**: 生产环境推荐

### buffalo_s

- **大小**: ~50 MB
- **精度**: 中等
- **速度**: 快
- **用途**: 快速测试

### buffalo_m

- **大小**: ~150 MB
- **精度**: 较高
- **速度**: 较快
- **用途**: 平衡选择

## 常见问题

### Q1: 下载总是中断怎么办？

**解决方案**：
1. 使用下载脚本（支持自动重试）
2. 多次运行（支持断点续传）
3. 检查网络稳定性
4. 使用代理或VPN

### Q2: 如何知道下载进度？

运行下载脚本会显示进度条。如果使用训练脚本，InsightFace 会自动显示下载进度。

### Q3: 下载的文件在哪里？

默认位置：
- **Linux/Mac**: `~/.insightface/models/buffalo_l/`
- **Windows**: `C:\Users\<用户名>\.insightface\models\buffalo_l\`

### Q4: 可以离线使用吗？

可以！下载完成后，模型文件会保存在本地，后续使用无需网络。

### Q5: 如何验证模型下载成功？

```bash
# 检查模型文件
python download_insightface.py --check

# 或手动检查
ls ~/.insightface/models/buffalo_l/
```

应该看到以下文件：
- `det_10g.onnx`
- `w600k_r50.onnx`
- `genderage.onnx`（可选）

## 代码中的改进

已更新 `utils.py` 中的 `get_insightface_detector` 函数：
- ✅ 添加了重试机制（默认3次）
- ✅ 更好的错误提示
- ✅ 支持断点续传（InsightFace 内置支持）

## 推荐做法

1. **首次使用**：运行下载脚本提前下载模型
   ```bash
   python download_insightface.py
   ```

2. **训练时**：如果模型已存在，会自动使用；如果不存在，会自动下载

3. **网络不稳定**：多次运行脚本或训练脚本，支持断点续传

