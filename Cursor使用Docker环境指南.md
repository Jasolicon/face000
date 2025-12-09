# Cursor 使用 Docker 环境指南

## 概述

Cursor（基于 VSCode）可以通过 **Remote-Containers** 扩展连接到 Docker 容器，直接在容器内进行开发和调试。这样可以在 Cursor 中直接使用 Docker 环境，无需手动进入容器。

## 前置要求

### 1. 安装 Docker

- **Windows/Mac**: 安装 [Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Linux**: 安装 Docker 和 nvidia-docker2（如果需要 GPU）

### 2. 安装 Remote-Containers 扩展

在 Cursor 中安装扩展：
- 打开扩展面板（`Ctrl+Shift+X` 或 `Cmd+Shift+X`）
- 搜索 "Remote - Containers"
- 安装 `ms-vscode-remote.remote-containers`

### 3. 构建 Docker 镜像

```bash
# 构建镜像
docker build -t face-gen:latest .

# 或者使用 devcontainer 专用 Dockerfile
docker build -f .devcontainer/Dockerfile -t face-gen:latest .
```

## 使用方法

### 方法 1: 使用 DevContainer（推荐）

#### 步骤 1: 打开项目

在 Cursor 中打开项目文件夹。

#### 步骤 2: 打开命令面板

- **Windows/Linux**: `Ctrl+Shift+P`
- **Mac**: `Cmd+Shift+P`

#### 步骤 3: 选择命令

输入并选择：
```
Remote-Containers: Reopen in Container
```

或者：
```
Dev Containers: Reopen in Container
```

#### 步骤 4: 等待容器启动

Cursor 会自动：
1. 构建/拉取 Docker 镜像（如果需要）
2. 启动容器
3. 安装扩展
4. 连接到容器

#### 步骤 5: 开始开发

连接成功后，Cursor 左下角会显示：
```
Dev Container: face-gen:latest
```

现在你可以在容器环境中直接编辑和运行代码了！

### 方法 2: 附加到运行中的容器

如果容器已经在运行：

1. 打开命令面板：`Ctrl+Shift+P` / `Cmd+Shift+P`
2. 选择：`Remote-Containers: Attach to Running Container`
3. 选择容器：`face-gen` 或 `face-generation-system`

## 配置说明

### `.devcontainer/devcontainer.json`

主要配置项：

```json
{
  "name": "多姿态人脸生成系统",
  "image": "face-gen:latest",  // 使用已构建的镜像
  
  // 或者自动构建
  "build": {
    "context": "..",
    "dockerfile": "../Dockerfile"
  },
  
  // 挂载目录
  "mounts": [
    "source=${localWorkspaceFolder}/features,target=/app/features",
    // ...
  ],
  
  // 环境变量
  "containerEnv": {
    "CUDA_VISIBLE_DEVICES": "0"
  },
  
  // VSCode 扩展
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance"
      ]
    }
  }
}
```

### 目录挂载

项目会自动挂载以下目录：

| 本地目录 | 容器目录 | 说明 |
|---------|---------|------|
| `./features` | `/app/features` | 特征数据 |
| `./features_arcface` | `/app/features_arcface` | ArcFace 特征 |
| `./generated_poses` | `/app/generated_poses` | 生成结果 |
| `./output` | `/app/output` | 输出文件 |
| `./input` | `/app/input` | 输入数据 |

## 功能特性

### 1. 直接在容器内运行代码

在 Cursor 中：
- 按 `F5` 运行 Python 文件
- 使用终端运行命令
- 使用调试器调试代码

### 2. Python 解释器

Cursor 会自动检测容器内的 Python：
- Python 路径：`/usr/local/bin/python`
- 已安装所有依赖
- 支持代码补全和类型检查

### 3. 终端

在容器内打开终端：
- **Windows/Linux**: `Ctrl+`` (反引号)
- **Mac**: `Cmd+``

终端会自动连接到容器，可以直接运行命令：

```bash
# 在容器内
python generate_multi_pose_faces.py --input /app/input/face.jpg
```

### 4. 扩展

容器内会自动安装以下扩展：
- Python
- Pylance（Python 语言服务器）
- Jupyter
- Docker

### 5. 调试

支持完整的调试功能：
- 设置断点
- 变量查看
- 调用堆栈
- 交互式调试

## 使用示例

### 示例 1: 运行脚本

1. 在 Cursor 中打开 `generate_multi_pose_faces.py`
2. 按 `F5` 或点击运行按钮
3. 选择 Python 解释器（容器内的）
4. 代码在容器内执行

### 示例 2: 使用终端

1. 打开终端（`Ctrl+``）
2. 运行命令：
   ```bash
   python test_file2.py
   ```

### 示例 3: 调试代码

1. 在代码中设置断点
2. 按 `F5` 启动调试
3. 使用调试面板查看变量和调用堆栈

## GPU 支持

### 检查 GPU

在容器终端中：

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 配置 GPU

在 `devcontainer.json` 中：

```json
{
  "runArgs": [
    "--gpus=all"
  ]
}
```

**注意**: 
- Windows/Mac Docker Desktop 需要启用 GPU 支持
- Linux 需要安装 nvidia-docker2

## 常见问题

### Q1: 无法连接到容器

**解决方案**:
1. 确保 Docker 正在运行
2. 检查镜像是否存在：`docker images | grep face-gen`
3. 如果不存在，先构建：`docker build -t face-gen:latest .`
4. 重新尝试连接

### Q2: 扩展未安装

**解决方案**:
1. 检查 `.devcontainer/devcontainer.json` 中的扩展列表
2. 手动安装扩展：在扩展面板搜索并安装
3. 重新加载窗口：`Ctrl+Shift+P` → `Developer: Reload Window`

### Q3: Python 解释器未检测到

**解决方案**:
1. 打开命令面板：`Ctrl+Shift+P`
2. 选择：`Python: Select Interpreter`
3. 选择容器内的 Python：`/usr/local/bin/python`

### Q4: 文件权限问题

**解决方案**:
在 `devcontainer.json` 中设置：

```json
{
  "remoteUser": "root"
}
```

### Q5: 模型下载慢

**解决方案**:
1. 使用镜像源（在容器内）：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```
2. 挂载本地缓存：
   ```json
   "mounts": [
     "source=${localEnv:HOME}/.cache/huggingface,target=/root/.cache/huggingface"
   ]
   ```

## 工作流程

### 开发流程

1. **打开项目** → Cursor 打开项目文件夹
2. **连接容器** → `Ctrl+Shift+P` → `Reopen in Container`
3. **等待连接** → 自动构建/启动容器
4. **开始开发** → 编辑、运行、调试代码
5. **保存文件** → 自动同步到容器

### 调试流程

1. **设置断点** → 在代码行号左侧点击
2. **启动调试** → 按 `F5` 或点击调试按钮
3. **查看变量** → 在调试面板查看
4. **继续执行** → 使用调试控制按钮

## 优势

### ✅ 环境一致性

- 所有开发者使用相同的环境
- 避免"在我机器上能跑"的问题
- 生产环境与开发环境一致

### ✅ 无需手动配置

- 自动安装依赖
- 自动配置 Python 环境
- 自动安装扩展

### ✅ 隔离性

- 不影响本地环境
- 可以同时运行多个项目
- 易于清理和重建

### ✅ 便捷性

- 直接在 IDE 中开发
- 完整的代码补全和调试
- 无需手动进入容器

## 快速开始

### 一键启动

1. 打开 Cursor
2. 打开项目文件夹
3. `Ctrl+Shift+P` → `Reopen in Container`
4. 等待连接完成
5. 开始开发！

### 验证环境

在容器终端中运行：

```bash
# 检查 Python
python --version

# 检查 PyTorch
python -c "import torch; print(torch.__version__)"

# 检查 CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

## 总结

- ✅ **Cursor 完全支持 Docker 环境**
- ✅ **通过 Remote-Containers 扩展连接**
- ✅ **直接在容器内开发和调试**
- ✅ **自动同步文件**
- ✅ **完整的 IDE 功能**

使用 Docker 环境可以让开发更加便捷和一致！

