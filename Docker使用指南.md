# Docker 使用指南

## 概述

本项目提供了完整的 Docker 环境配置，基于 **PyTorch 2.2.2 + CUDA 11.8**，可以直接使用，无需手动配置环境。

## 前置要求

### 1. 安装 Docker

#### Linux
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装 nvidia-docker2（GPU 支持）
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

#### Windows
1. 安装 [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
2. 安装 [WSL 2](https://docs.microsoft.com/windows/wsl/install)
3. 在 Docker Desktop 中启用 WSL 2 后端

#### macOS
1. 安装 [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop)
2. 注意：macOS 不支持 GPU，只能使用 CPU 版本

### 2. 验证安装

```bash
# 检查 Docker
docker --version

# 检查 GPU 支持（Linux）
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## 快速开始

### 方法 1: 使用 Docker Compose（推荐）

```bash
# 构建镜像
docker-compose build

# 启动容器
docker-compose up -d

# 进入容器
docker-compose exec face-gen bash

# 在容器内运行脚本
python generate_multi_pose_faces.py --input /app/input/face.jpg --output-dir /app/output
```

### 方法 2: 使用 Docker 命令

```bash
# 构建镜像
docker build -t face-gen:latest .

# 运行容器（GPU）
docker run --gpus all -it --rm \
  -v $(pwd)/features:/app/features \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/input:/app/input:ro \
  face-gen:latest \
  bash

# 在容器内运行
python generate_multi_pose_faces.py --input /app/input/face.jpg --output-dir /app/output
```

## 镜像说明

### 基础镜像

```
pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime
```

- **PyTorch**: 2.2.2
- **CUDA**: 11.8
- **cuDNN**: 8
- **Python**: 3.10（默认）

### 已安装的依赖

- torch==2.2.2（基础镜像自带）
- torchvision==0.17.2
- 所有 requirements.txt 中的依赖

## 目录结构

### 容器内目录

```
/app/
├── features/          # 特征存储目录
├── features_arcface/  # ArcFace 特征目录
├── generated_poses/   # 生成的多姿态图像
├── output/           # 输出目录
└── input/            # 输入数据目录（只读）
```

### 挂载映射

| 容器路径 | 主机路径 | 说明 |
|---------|---------|------|
| `/app/features` | `./features` | 特征数据 |
| `/app/features_arcface` | `./features_arcface` | ArcFace 特征 |
| `/app/generated_poses` | `./generated_poses` | 生成结果 |
| `/app/output` | `./output` | 输出文件 |
| `/app/input` | `./input` | 输入数据（只读） |
| `~/.cache/huggingface` | `~/.cache/huggingface` | 模型缓存 |

## 使用示例

### 示例 1: 生成多姿态人脸

```bash
# 进入容器
docker-compose exec face-gen bash

# 运行生成脚本
python generate_multi_pose_faces.py \
  --input /app/input/test_face.jpg \
  --output-dir /app/output \
  --poses side down up left right
```

### 示例 2: 批量处理

```bash
# 在容器内
python batch_process.py \
  --dirs /app/input/person1 /app/input/person2 \
  --use-arcface \
  --storage-dir /app/features_arcface
```

### 示例 3: 视频帧处理

```bash
# 在容器内
python test_file2.py
```

## GPU 支持

### 检查 GPU

```bash
# 在容器内
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### 使用 GPU

确保：
1. 安装了 nvidia-docker2（Linux）
2. 使用 `--gpus all` 参数（Docker 命令）
3. 或使用 `runtime: nvidia`（docker-compose）

## CPU 版本

如果没有 GPU，可以使用 CPU 版本：

### 修改 Dockerfile

```dockerfile
# 使用 CPU 版本的基础镜像
FROM pytorch/pytorch:2.2.2-cpu

# 安装 CPU 版本的 torchvision
RUN pip install torchvision==0.17.2+cpu --index-url https://download.pytorch.org/whl/cpu
```

### 构建 CPU 版本

```bash
docker build -f Dockerfile.cpu -t face-gen:cpu .
```

## 常见问题

### Q1: 容器内无法使用 GPU

**解决方案**:
1. 检查 nvidia-docker2 是否安装
2. 检查 Docker 是否支持 GPU：
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```
3. 确保使用 `--gpus all` 或 `runtime: nvidia`

### Q2: 模型下载慢

**解决方案**:
1. 使用镜像源：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```
2. 挂载本地模型缓存：
   ```yaml
   volumes:
     - ~/.cache/huggingface:/root/.cache/huggingface
   ```

### Q3: 容器内文件权限问题

**解决方案**:
```bash
# 在 Dockerfile 中设置权限
RUN chmod -R 755 /app

# 或在运行时指定用户
docker run --user $(id -u):$(id -g) ...
```

### Q4: 显存不足

**解决方案**:
1. 使用 FP16：
   ```python
   generator = MultiPoseFaceGenerator(use_fp16=True)
   ```
2. 减少推理步数：
   ```python
   num_inference_steps=10  # 默认 20
   ```
3. 限制 GPU 使用：
   ```yaml
   environment:
     - CUDA_VISIBLE_DEVICES=0  # 只使用第一个 GPU
   ```

### Q5: 容器启动慢

**解决方案**:
1. 使用预构建镜像（如果有）
2. 挂载模型缓存目录
3. 使用多阶段构建优化镜像大小

## 优化建议

### 1. 使用多阶段构建

```dockerfile
# 构建阶段
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# 运行阶段
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
```

### 2. 缓存模型

```yaml
volumes:
  - ~/.cache/huggingface:/root/.cache/huggingface
```

### 3. 使用 .dockerignore

已创建 `.dockerignore` 文件，排除不必要的文件以加快构建。

## 生产环境部署

### 1. 使用 Docker Compose

```bash
# 后台运行
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止
docker-compose down
```

### 2. 使用 Kubernetes

创建 Kubernetes 部署文件（需要 GPU 节点）：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: face-gen
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: face-gen
        image: face-gen:latest
        resources:
          limits:
            nvidia.com/gpu: 1
```

## 镜像构建

### 构建镜像

```bash
# 标准构建
docker build -t face-gen:latest .

# 指定平台（ARM）
docker build --platform linux/arm64 -t face-gen:arm64 .

# 不使用缓存
docker build --no-cache -t face-gen:latest .
```

### 推送镜像

```bash
# 标记镜像
docker tag face-gen:latest your-registry/face-gen:latest

# 推送
docker push your-registry/face-gen:latest
```

## 环境变量

可以在 `docker-compose.yml` 中设置环境变量：

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0
  - HF_ENDPOINT=https://hf-mirror.com
  - TORCH_HOME=/app/models
```

## 总结

- ✅ **完整的 Docker 配置**：Dockerfile + docker-compose.yml
- ✅ **GPU 支持**：基于 CUDA 11.8
- ✅ **版本固定**：torch 2.2.2 + torchvision 0.17.2
- ✅ **数据持久化**：通过 volumes 挂载
- ✅ **易于使用**：一键启动和运行

使用 Docker 可以确保环境一致性，避免版本冲突问题！

